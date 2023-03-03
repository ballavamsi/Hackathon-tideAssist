# tideAssist

This is an assistant created powered by OpenAI to get answers to questions with data from the TIDE website.
For now we are just ingesting few pages which can be found in ./data folder

## How to run

```
pip install -r requirements.txt
python app.py
```

It opens a webserver on port 7173. You can access it on http://localhost:7173
I have created UI using gradio to demonstrate the app.

## How to train

```
Update the content in ./data folder
When you rerun the app, it will automatically train the model
```

## How to test

Open webserver on http://localhost:7173
Enter the question in the text box
The AI will generate answer the questions.

## Where can i use this?

This can be used as assistant in tide website. There can be 2 phases to this.
The first phase is to use generate answers to questions for prospect clients in getting answers and making them chose best plan.
The second phase could be to get post login clients to ask questions about their balances or uploading receipts or any other thing.

## Sample screens

### Screenshot

![alt text](.\media\sample.png "Sample screen")

### Demo

![alt text](.\media\demo.gif "Gif sample")
