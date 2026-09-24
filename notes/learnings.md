#Sentence Embedding:
Sentence Embedding is converting the sentences into a list of numbers/vectors which captures its meaning. In this, The no. of vectors are placed in a vector space checking its similarities and more they are close, more the chances of similarities

Cosine Similarity:
Cosine Similarity is the Formula in which the angle btw the vectors are calculated and if both the vectors are pointed at a similiar direction, then the possibility of similarities btw the victors will became high.
            Cosine Similarity: A.B/ ||A||.||B||

Working:
1. The vector model will convert the sentences into vectors.
2. The vector of diff models are compared and angle btw them is calculated using the cosine similarity.
3. The more they are close, the more the chances of similarity.

*This process is better than the keyword matching.

#Parser.py Code:
This code takes a Pdf resume, opens it, extracts all the words from all the pages, combines all the text into no. string and returns it.
Path: Helps Python to work with file and folders
sys: Gives access to Python/system level functionality
*Pdfplumber: opens a pdf and extract its text
. Main Function: Open the Pdf-> Extract the text to the file path-> Return the String
. Convert the File Path into File object so that python can work clearly
. Create an empty list to store the extracted text.
. Pdfplumber will open every page-> Loop through Every Page-> Extract text from the page
. If there is no words or blank space btw the pages so the python will *strip(Remove the space) and continues
. Append to the list-> Combine all the pages-> Joins into one String
. Funcion will return the String-> Which can be used for further embeddings

#Embedder.py Code:
This code takes 2 pieces of text and convert it into numerical embeddings using sentence transformer and then check the similarity btw the two texts using cosine similarity.
              Pointing in similiar direction-> High similarity
              Pointing in different direction-> Low similarity
Path: To locate sample resume and Job Description
Sentence Transformer: Pre-Trained sentence transformer model to convert text into embeddings
Cosine Similarity: To check the similarity   
Sentence Embedding-> Create 1st and 2nd-> Compute Similarity
*Item: To extract actual Python no.
*Clipping: Intention to make the final value stays btw 0 and 1 -> Returns in Float[0,1]
Compares the pdf with the JD-> extract text-> Create Embeddings-> Check the similarity Score-> Print the Score

Results:
Resume 1-> 0.717
Resume 2-> 0.379
Resume 3-> 0.36
Why the diff btw Resume 2(Data Analyst) and Resume 3(Customer Service) is lower because its based on semantic embedding btw the resume and jd not keywords and they both have differnet semantic content

#Keyword_Analysis:
This analyze the skills mention in the jd and in the resume and checks the similarities and the missing ones.
re-> regular expression -> used for pattern matching and text cleaning
Curated_List-> It doesn't allow duplicate skills
Skill variants-> It takes one skill and checks how this can be written in different ways
Find_Missing_Keywords-> It is used to find a list of skills
It compares the curated list of skills to the skills that are required for the JD and rest of the skills are termed as missing keywords




