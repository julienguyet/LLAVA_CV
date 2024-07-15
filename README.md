# Fine-tuning a Multimodal Model for Healthcare

---

## 1. Dataset :stethoscope:

The data used to fine-tuned the multimodal model is the [Path-VQA dataset](https://arxiv.org/abs/2003.10286). It is made of 30k image-question-answer triplets and is freely available on HuggingFace [here](https://huggingface.co/datasets/flaviagiammarino/path-vqa).

In this project, we have also trained YOLOv8 on a brain tumor dataset available on [Kaggle](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation). We have done this as our aim is to compare the performance of a "classic" ML model for computer vision and a VLM.

---

## 2. Vision Language Model :volcano:

In this project, we have fine-tuned LLaVA, a multimodal model combining the power of CLIP and Vicuna. More information are available on the official [website](https://llava-vl.github.io/). We loaded and trained the model using the model card from HuggingFace (see [here](https://huggingface.co/liuhaotian/llava-v1.5-13b)).

Experiments and detailled results are available in the dedicated LLAVA_fine_tuned notebook. 

---

## 3. Results :checkered_flag:

With a training on only 1200 samples from the train set and on 3 epochs, we already observed strong performance from the model, with an accuracy of 67%. That said, additional research is needed to confirm those results: due to its verbosity, the model sometimes fails to be concise and therefore its answers can be quite different from the correct answers from the dataset. However, after manual review, some generated outputs appears to be quite correct but rephrased in a total different way. This highlight the need to involve physicians in the evaluating process when fine-tuning VLMs on such datasets.

---

## 4. What if I want to use this model? :technologist:

The parameters of the fine-tuned model are available in the fine_tuned_model folder. If you wish to test this model on your notebook you can load it by executing the below code:

```python
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

model = LlavaForConditionalGeneration.from_pretrained(saved_model_path,
    torch_dtype=torch.float16,
    quantization_config=quantization_config)
```

Then you can call the model following the format from LLAVA documentation:

```python
item = test_data[int(index)]
image = item['image']
question = item['question']
answer = item['answer']
prompt = f"USER: <image>\n{question}\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt")
generate_ids = model.generate(**inputs, max_new_tokens=100)
answer = processor.batch_decode(base_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(answer)
```
