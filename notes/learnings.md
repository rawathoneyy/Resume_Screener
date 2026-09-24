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

    

