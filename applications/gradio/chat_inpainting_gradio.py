import os
import gradio as gr
import numpy as np
import uuid
import shutil
import paddle
import paddle.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from paddlevlp.models.blip2.modeling import Blip2ForConditionalGeneration
from paddlevlp.processors.blip_processing import Blip2Processor
from paddlevlp.processors.groundingdino_processing import GroudingDinoProcessor
from paddlevlp.models.groundingdino.modeling import GroundingDinoModel
from paddlevlp.models.sam.modeling import SamModel
from paddlevlp.processors.sam_processing import SamProcessor
from ppdiffusers import StableDiffusionInpaintPipeline

from paddlenlp import Taskflow

from paddlevlp.utils.log import logger

def generate_caption(raw_image, processor,blip2_model):
    
    inputs = processor(
        images=raw_image,
        text=None,
        return_tensors="pd",
        return_attention_mask=True,
        mode="test",
    )
    generated_ids, scores = blip2_model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()
    logger.info("Generate text: {}".format(generated_text))

    return generated_text

class ConversationBot:
    def __init__(self):

        logger.info("stable diffusion pipeline: {}".format("stabilityai/stable-diffusion-2-inpainting"))
        self.sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")
        logger.info("stable diffusion pipeline build finish!")
    
        logger.info("dino_model: {}".format("GroundingDino/groundingdino-swint-ogc"))
        #bulid dino processor
        self.dino_processor = GroudingDinoProcessor.from_pretrained(
        "GroundingDino/groundingdino-swint-ogc"
        ) 
        #bulid dino model
        self.dino_model = GroundingDinoModel.from_pretrained("GroundingDino/groundingdino-swint-ogc")
        self.dino_model.eval()
        logger.info("dino_model build finish!")

        #buidl sam processor
        self.sam_processor = SamProcessor.from_pretrained(
            "Sam/SamVitH-1024"
        ) 
        #bulid model
        logger.info("SamModel: {}".format("Sam/SamVitH-1024"))
        self.sam_model = SamModel.from_pretrained("Sam/SamVitH-1024",input_type="boxs")
        logger.info("SamModel build finish!")

        logger.info("chatglm: {}".format("THUDM/chatglm-6b"))
        self.textGen = Taskflow("text2text_generation",model="THUDM/chatglm-6b")

        logger.info("blip2_model: {}".format("Salesforce/blip2-opt-2.7b"))
        #bulid blip2 processor
        self.blip2_processor = Blip2Processor.from_pretrained(
           "Salesforce/blip2-opt-2.7b"
        )  # "Salesforce/blip2-opt-2.7b"
        #bulid blip2 model
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

        self.blip2_model.eval()
        self.blip2_model.to("gpu")
        logger.info("blip2_model build finish!")

        self.image_pil = None
        self.det_prompt = None
        self.inpaint_prompt = None
        self.image_filename = None
    
    def run_text(self,text,state):
        warning_prompt = None

        if self.image_pil is None:
           warning_prompt = 'Please upload image'
        if self.det_prompt is None or self.inpaint_prompt is None:
           warning_prompt = 'Please input prompt'
        if warning_prompt is not None:
            state = state + [(warning_prompt,"exp:Replace the dog with a cat")]

            return state,state

        prompt = "Given caption,extract the main object to be replaced and marked it as 'main_object', " + \
              f"Extract the remaining part as 'other prompt', " + \
              f"Return main_object, other prompt in English" + \
              f"Given caption: {text}."

        reply = self.textGen(prompt)['result'][0]
     
        self.det_prompt,self.inpaint_prompt = reply.split('\n')[0].split(':')[-1].strip(), reply.split('\n')[-1].split(':')[-1].strip()
        
        logger.info("det prompt: {}".format(self.det_prompt))
        logger.info("inpaint prompt: {}".format(self.inpaint_prompt))
        
        AI_prompt = "Received.  "
        state = state + [(text,AI_prompt)]
        self.run()
        state = state + [(f"![](/file={self.image_filename})","Done!")]

        return state,state
    def run_image(self, image, state):
        suffix = image.name.split('.')[-1]
        self.image_filename = image.name.split('.')[0]+"_chatglm_output."+suffix
        self.image_pil = Image.open(image.name)
        caption = generate_caption(self.image_pil,processor=self.blip2_processor,blip2_model=self.blip2_model)
        state = state + [(caption,f"![](/file={image.name})")]
        
        return state,state

    def run(self):
        image_pil = self.image_pil.convert("RGB")

        #preprocess image text_prompt
        image_tensor,mask,tokenized_out = self.dino_processor(images=image_pil,text=self.det_prompt)

        with paddle.no_grad():
            outputs = self.dino_model(image_tensor,mask, input_ids=tokenized_out['input_ids'],
                            attention_mask=tokenized_out['attention_mask'],text_self_attention_masks=tokenized_out['text_self_attention_masks'],
                            position_ids=tokenized_out['position_ids'])

        logits = F.sigmoid(outputs["pred_logits"])[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(axis=1) > 0.3
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = self.dino_processor.decode(logit > 0.25)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        logger.info("dino output{}".format(pred_dict))
        
        H,W = size[1], size[0]
        boxes = []
        for box in zip(boxes_filt):
            box = box[0] * paddle.to_tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box.numpy()
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            boxes.append([x0, y0, x1, y1])
        boxes = np.array(boxes)
        image_seg,prompt = self.sam_processor(image_pil,input_type="boxs",box=boxes,point_coords=None) 
        seg_masks = self.sam_model(img=image_seg,prompt=prompt)
        seg_masks = self.sam_processor.postprocess_masks(seg_masks)

        logger.info("Sam finish!")

        merge_mask = paddle.sum(seg_masks,axis=0).unsqueeze(0)
        merge_mask = merge_mask>0
        mask_pil = Image.fromarray(merge_mask[0][0].cpu().numpy())
    
        image_pil = image_pil.resize((512, 512))
        mask_pil = mask_pil.resize((512, 512))
    
        image = self.sd_pipe(prompt=self.inpaint_prompt, image=image_pil, mask_image=mask_pil).images[0]
        image = image.resize(size)
        image.save(self.image_filename)
    
    def clear(self):
        self.image_pil = None
        self.det_prompt = None
        self.inpaint_prompt = None
        self.image_filename = None
       


bot = ConversationBot()
with gr.Blocks(css="#chatInpainting {overflow:auto; height:500px;}") as demo:
        gr.Markdown("<h3><center>ChatInpainting</center></h3>")
        gr.Markdown(
            """This is a demo to the work [PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX).<br>
            """
        )


        chatbot = gr.Chatbot(elem_id="chatbot", label="ChatBot")
        state = gr.State([])

        with gr.Row(visible=True) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(container=False)
      
            with gr.Column(scale=0.10, min_width=0):
                btn = gr.UploadButton("🖼️Upload", file_types=["image"])
            with gr.Column(scale=0.10, min_width=0):
                clear = gr.Button("🔄Clear️")
        
        
        btn.upload(bot.run_image, [btn, state], [chatbot, state])
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)

        clear.click(bot.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)

        

demo.launch(share=True)   


