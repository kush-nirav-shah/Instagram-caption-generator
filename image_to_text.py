from langchain_community.llms.ctransformers import CTransformers
from langchain_core.output_parsers import StrOutputParser
from langchain import PromptTemplate
from huggingface_hub import login
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import matplotlib.pyplot as plt

login("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")

model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
llm=CTransformers(model=model,
                  model_file=model_file,
                  config={'max_new_tokens':512,
                          'temperature':0.6,
                          'context_length':1024,
                          'gpu_layers':22})



model = VisionEncoderDecoderModel.from_pretrained('vit-gpt2-image-captioning')
feature_extractor = ViTImageProcessor.from_pretrained(
    'vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('vit-gpt2-image-captioning')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {'max_length': max_length, 'num_beams': num_beams}

def predict_step(image):
    pixel_values = feature_extractor(
        images=[image], return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]

image = Image.open(input("Enter Path"))
image_to_text = predict_step(image=image)

prompt_template= """
Give an interesting one line social media caption for the given text: {context}.
Be creative and you may add hashtags and emojis if any.
Caption:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
parser = StrOutputParser()
chain = prompt | llm | parser
chain

response = chain.invoke(image_to_text)

plt.imshow(image)
plt.axis('off')  # Hide axis
plt.show()

print(response)
