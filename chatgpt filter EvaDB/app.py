import os 
import evadb

os.environ["OPENAI_KEY"] = "API"
print(os.environ["OPENAI_KEY"])
cursor = evadb.connect().cursor()
cursor.drop_table("MyPDF").execute()
cursor.load(file_regex="./layout-parser-paper.pdf",
            format="pdf", table_name="MyPDF").execute()
df = cursor.table("MyPDF").df()
print(df)
#cursor.drop_function("ChatGPT").df()
#cursor.create_function("ChatGPT",True,"./chatgpt.py").df()

question = "Act as a text classification model that classifies paragraphs extracted from research papers. Classify front-matter, reference citations, and table of contents as 'noise'. Classify the body of the paper as 'signal'.  Classify this text fragment:"
column = "text"
prompt = "Return only one word in your response 'noise' or 'signal'."

import pandas as pd
pd.set_option('display.max_colwidth', None)

output = cursor.table("MyPDF").filter("page > 2 AND page < 4").select(f"""ChatGPT("{question}", data, "{prompt}")""").df()

output.to_csv("output.csv")

print(output)