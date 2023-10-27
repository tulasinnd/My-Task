import PyPDF2
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
summary_model = T5ForConditionalGeneration.from_pretrained('t5-small')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)
import os
os.environ["OPENAI_API_KEY"] ='Replace_API_KEY'
from langchain.llms import OpenAI

# converts the pdf file to list of paragraphs
def pdf_to_paragraphs_list(file_path):
    topics_list=[]
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            topics_list.append(text)
    return topics_list

# summarizes the list of paragraphs to simple sentences
def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, 
                                   pad_to_max_length=False,
                                   truncation=True, 
                                   return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
  outs = model.generate(input_ids=input_ids,
                        attention_mask=attention_mask,
                        early_stopping=True,
                        num_beams=3,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        min_length = 80,
                        max_length=250)
  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary= summary.strip()
  return summary

# by using context multiple choice questions are generated using openai api
def get_mca_questions(context):
    if not isinstance(context, str):
        raise TypeError("The argument must be of string data type.") 
    mca_questions = []
    large_language_model = OpenAI(temperature=0.9)  # model_name="text-davinci-003"
    text = '''  step 1: understand the content
                step 2: generate 5 multiple choice questions with four options a, b, c, d
                step 3: in each question, there should be multiple correct options. 
                        User should be able to choose any two options as correct 
                        answers based on the information provided in the context.
                step 4: mark the correct answers with *CORRECT* at the end of option
                step 5: while generating questions, dont write any titles or side headings
                content is :  '''+context
    mcq=large_language_model(text)
    mca_questions=mcq.split('\n\n')
    print(mcq)
    return mca_questions

# convert pdf file to paragraph and summarise the text
file_path =r"chapter-4.pdf"
paragraphs_list = pdf_to_paragraphs_list(file_path) 
paragraphs_summary_list=[]
for i in paragraphs_list:
    summ=summarizer(i,summary_model,summary_tokenizer)
    paragraphs_summary_list.append(summ)

# all the small summaries are combined to 5000 tokens
final_summary = ' '.join(paragraphs_summary_list)
if len(final_summary)>5000:
    final_summary=final_summary[:5000]

# get multiple question with more than one correct answer by passing context
mca=get_mca_questions(final_summary)
for question in mca:
    print(question)
    print()