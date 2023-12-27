# Libraries 
import customtkinter as ctk
import tkinter as tk
from time import sleep
import threading
from PIL import Image,ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import re
import nltk
import os

#Global values
Logo = ctk.CTkImage(dark_image= Image.open("Images/Logo.png"), size=(200,100))
DarkLogo = ctk.CTkImage(dark_image= Image.open("Images/LogoDark.png"), size=(200,100))
#Light and dark mode
Basic_Font_ColorL = "#000000"
Basic_Background_ColorL = "#FFFFFF"
DarkMode = ctk.CTkImage(dark_image= Image.open("Images/darkmode.png"), size=(20,20))
LightMode = ctk.CTkImage(dark_image= Image.open("Images/brightness.png"), size=(20,20))
Basic_Font_ColorD = "#FFFFFF"
Basic_Background_ColorD = "#000000"
search_path = "Models/file.txt"
default_font = "Roboto"
default_font_size = 15
Score = 21
sentiment_pos = 0.0

#Time

class MyFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # add widgets onto the frame, for example:
        self.label = ctk.CTkLabel(self,text='')
        self.label.grid(row=0, column=0, padx=0)

class StartWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.configure(fg_color = Basic_Background_ColorL)
        self.iconbitmap("Images/Icon.ico")
        self.geometry("340x315")# size
        self.title("Review Estimaor Start")#title

        self.banner = ctk.CTkLabel(self, image = Logo, text ="")
        self.banner.grid(row = 1, column = 1, columnspan = 3,sticky = "ew")
        


        self.Message = ctk.CTkLabel (self, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD), text_color= (Basic_Font_ColorL, Basic_Font_ColorD),font = ("Roboto",15) ,text = "                                  ")
        self.Message.grid(row = 3, column = 2,sticky = "ew", padx = 10, pady =10)
        
        self.progressbar = ctk.CTkProgressBar(master=self,mode = "determinate", determinate_speed = 1 )
        self.progressbar.grid(row = 2, column = 2,sticky = "ew", padx = 10, pady =10)

        self.progressbar.set(0)
    
        self.Startbutton = ctk.CTkButton(self,fg_color = (Basic_Background_ColorL, Basic_Background_ColorD), text_color= (Basic_Font_ColorL, Basic_Font_ColorD), height= 10, width = 10 , text="Start", command =self.start_check)
        self.Startbutton.grid(row = 4, column = 2, sticky = "ew", padx = 5, pady = 5)
        
        self.modebutton = ctk.CTkButton(self, image = LightMode, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD), height= 10, width = 10 , text="", command =self.changemode)
        self.modebutton.grid(row = 3, column = 0, sticky = "sw", padx = 5, pady = 5)

        self.Exitbutton = ctk.CTkButton(self,fg_color = (Basic_Background_ColorL, Basic_Background_ColorD), text_color= (Basic_Font_ColorL, Basic_Font_ColorD), height= 10, width = 10 , text="Exit", command =self.exitroot)
        self.Exitbutton.grid(row = 6, column = 2, sticky = "ew", padx = 5, pady = 5)
    
    def exitroot(self):
        self.destroy()
    
    def bar(self):
        n = 500
        iter_step = 1/n
        progress_step = iter_step
        self.progressbar.start()
        
        for x in range(500):
            self.progressbar.set(progress_step)
            progress_step += iter_step
            self.update_idletasks()
        self.progressbar.stop()

    def changemode(self):
        if self.modebutton.cget("image") == LightMode:
            self.modebutton.configure(image=DarkMode, fg_color = Basic_Background_ColorD)
            ctk.set_appearance_mode("dark")
            self.banner.configure(image = DarkLogo)
            self.configure(fg_color = Basic_Background_ColorD)
        
        else:
            self.modebutton.configure(image=LightMode, fg_color = Basic_Background_ColorL)
            ctk.set_appearance_mode("light")
            self.banner.configure(image = Logo)
            self.configure(fg_color = Basic_Background_ColorL)
    def start_check(self):
    
        if os.path.isfile('./model_log_reg') == True and os.path.isfile('./model_tfidf') == True:
            self.Message.configure(text = "The model number 1 is saved")
            sleep(2)
        else:
            semanlysis_dataset = pd.read_csv("IMDB Dataset.csv")
            # Split the data
            train, test = train_test_split(semanlysis_dataset, test_size=0.25, train_size=.75, random_state=42)

            # Define features and labels
            train_x, train_y = train['review'], train['sentiment']
            test_x, test_y = test['review'], test['sentiment']

            # TF-IDF vectorization
            tfidf = TfidfVectorizer(stop_words='english')
            train_x_vector = tfidf.fit_transform(train_x)
            test_x_vector = tfidf.transform(test_x)

            with open ('model_tfidf', 'wb') as files:
                pickle.dump(tfidf, files)
            # Define and train the model
            log_reg = LogisticRegression()
            log_reg.fit(train_x_vector, train_y)

            # Evaluate the model
            accuracy = log_reg.score(test_x_vector, test_y)
            print(f"Accuracy: {accuracy}")

            # Make predictions
            prediction = log_reg.predict(tfidf.transform(['I did not like this movie at all']))
            print(prediction)


            with open ('model_log_reg', 'wb') as files:
                pickle.dump(log_reg, files)
            self.Message.configure(text = "The model number 1 has been saved")
        n = 500
        iter_step = 1/n
        progress_step = iter_step
        self.progressbar.start()
        for x in range(250):
            self.progressbar.set(progress_step)
            progress_step += iter_step
            self.update_idletasks()
        self.progressbar.stop()
        if os.path.isfile('./model_sentiment_values') == True:
            self.Message.configure(text = "The model number 2 is saved")
            sleep(2)
        else:
            # Load the data
            rotten_tomatoes_reviews_adjusted = pd.read_csv('Rotten_Tomato_Sentiment.csv')

            # Drop rows with missing values in relevant columns
            rotten_tomatoes_reviews_adjusted = rotten_tomatoes_reviews_adjusted.dropna(subset=['sentiment', 'neg_count', 'pos_count', 'total_len', 'predicted_sentiment', 'review_score'])

            # Split the data
            train, test = train_test_split(rotten_tomatoes_reviews_adjusted, test_size=0.25, train_size=0.75, random_state=42)

            # Define features and labels
            feature_cols = ['sentiment', 'neg_count', 'pos_count', 'total_len', 'predicted_sentiment']
            train_y = train['review_score']
            test_y = test['review_score']

            # Define additional features
            train_x_other = train[feature_cols].reset_index(drop=True)
            test_x_other = test[feature_cols].reset_index(drop=True)

            # Define and train the model (using Linear Regression for regression task)
            linear_reg = LinearRegression()
            linear_reg.fit(train_x_other, train_y)

            # Evaluate the model
            predictions = linear_reg.predict(test_x_other)
            mse = mean_squared_error(test_y, predictions)
            print(f"Mean Squared Error: {mse}")

            with open ('model_sentiment_values', 'wb') as files:
                    pickle.dump(linear_reg, files)
            self.Message.configure(text = "The model number 2 has been saved")
        self.progressbar.start()
        for x in range(250):
            self.progressbar.set(progress_step)
            progress_step += iter_step
            self.update_idletasks()
        self.progressbar.stop()
        MainWindow(self)
        self.withdraw()
    """
    def threading_start(self):
        # Start the start_check thread
        start_check_thread = threading.Thread(target=self.start_check, name='start_check_thread')
        start_check_thread.start()

        # Schedule the bar method to run after the start_check method completes
        self.after(100, self.run_bar_thread)

    def run_bar_thread(self):
        # Start the bar thread
        bar_thread = threading.Thread(target=self.bar, name='bar_thread')
        bar_thread.start()

        # Wait for the bar thread to complete
        bar_thread.join()

        # After the bar thread is done, show the MainWindow
        MainWindow(self)
        self.withdraw()

    """
        


        
class MainWindow(ctk.CTkToplevel):
    def __init__(self , master):
        super().__init__(master)
        self.after(250, lambda: self.iconbitmap('Images/Icon.ico'))
        self.geometry("800x600")# size
        self.title("Review Estimaor Start")#title

        self.banner = ctk.CTkLabel(self, image = Logo, text ="")
        self.banner.grid(row = 0, column = 0, columnspan = 3,sticky = "nw")

        self.modebutton = ctk.CTkButton(self, image = LightMode, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD), height= 10, width = 10 , text="", command =self.changemode)
        self.modebutton.grid(row = 0, column = 15, sticky = "ne", padx = 5, pady = 5)

        self.leftframe = MyFrame(self, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD))
        self.leftframe.grid(row = 1, column = 0, columnspan = 2, rowspan = 2,sticky = "w", padx = 10, pady =30)

        self.LabelComment = ctk.CTkLabel(self.leftframe, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD ),font = (default_font,default_font_size)  , text = "Enter Comment -")
        self.LabelComment.grid(row = 0, column = 0,sticky = "nw", padx = 10, pady =10)

        #self.Comment= ctk.StringVar()
        self.text_area = tk.Text(self.leftframe, wrap=tk.WORD, 
                                      width=40, height=8, 
                                      font=(default_font, default_font_size)) 
        #self.enterComment = ctk.CTkEntry (self.leftframe, width = 60, height = 50, textvariable=self.Comment,font = (default_font,default_font_size)  )
        self.text_area.grid(row = 1, column = 0, columnspan = 8,sticky = "nswe", padx = 10, pady =30)

        
        self.estimationbutton = ctk.CTkButton(self.leftframe, text = "Estimate", fg_color = (Basic_Font_ColorL, Basic_Font_ColorD),text_color= (Basic_Background_ColorL, Basic_Background_ColorD) ,command = self.estimate)
        self.estimationbutton.grid(row = 4, column = 1, columnspan = 1,sticky = "w", padx = 20, pady =20)

        self.clearbutton = ctk.CTkButton(self.leftframe, text = "Clear", fg_color = (Basic_Font_ColorL, Basic_Font_ColorD),text_color= (Basic_Background_ColorL, Basic_Background_ColorD) ,command = self.clear)
        self.clearbutton.grid(row = 4, column = 0, columnspan = 1,sticky = "w", padx = 20, pady =20)

        self.rightframe = MyFrame(self, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD))
        self.rightframe.grid(row = 0, column = 10, columnspan = 2, rowspan = 2,sticky = "e", padx = 10, pady =30)

        self.sentimentvar = ctk.CTkLabel (self.rightframe, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD ),font = (default_font,default_font_size)  , text = "Sentiment -",)
        self.sentimentvar.grid(row = 1, column = 0, columnspan = 1,sticky = "ew", padx = 10, pady =10)

        self.sentimentaccuracy = ctk.CTkLabel (self.rightframe, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD ),font = (default_font,default_font_size)  , text = "Sentiment Accuracy -",)
        self.sentimentaccuracy.grid(row = 2, column = 0, columnspan = 1,sticky = "ew", padx = 10, pady =10)

        self.sentimentscore = ctk.CTkLabel (self.rightframe, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD ),font = (default_font,default_font_size)  , text = "Sentiment Score - ",)
        self.sentimentscore.grid(row = 3, column = 0, columnspan = 1,sticky = "ew", padx = 10, pady =10)

        self.length = ctk.CTkLabel (self.rightframe, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD ),font = (default_font,default_font_size)  , text = "Number of Useful Words - ",)
        self.length.grid(row = 4, column = 0, columnspan = 1,sticky = "ew", padx = 10, pady =10)

        self.num_positive = ctk.CTkLabel (self.rightframe, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD ),font = (default_font,default_font_size)  , text = "Number of Positive Words -",)
        self.num_positive.grid(row = 5, column = 0, columnspan = 1,sticky = "ew", padx = 10, pady =10)

        self.num_negative = ctk.CTkLabel (self.rightframe, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD ),font = (default_font,default_font_size)  , text = "Number of Negative Words -",)
        self.num_negative.grid(row = 6, column = 0, columnspan = 1,sticky = "ew", padx = 10, pady = 10)

        self.scoreoptions = ctk.CTkSegmentedButton(self.rightframe, values=["/20", "/5", "/4", "Percent"], command=self.score_options)
        self.scoreoptions.set("/20")
        self.scoreoptions.grid(row = 7, column = 0, columnspan = 1,sticky = "ew", padx = 10, pady =10)

        self.Score = ctk.CTkLabel (self.rightframe, fg_color = (Basic_Background_ColorL, Basic_Background_ColorD ),font = (default_font,default_font_size)  , text = "Score - ",)
        self.Score.grid(row = 8, column = 0, columnspan = 1,sticky = "ew", padx = 10, pady =10)

        self.XRexitbutton = ctk.CTkButton(self.rightframe, text = "Exit", fg_color = (Basic_Font_ColorL, Basic_Font_ColorD),text_color= (Basic_Background_ColorL, Basic_Background_ColorD) ,command = self.exitWindow)
        self.XRexitbutton.grid(row = 9, column = 0, columnspan = 1,sticky = "ew", padx = 20, pady =20)

    def score_options(self, choice):
        global Score 
        if Score == 21:
            self.Score.configure(text = f"Score -")
        elif choice == "/20":
            self.Score.configure(text = f"Score - {Score}/20")
        elif choice == "/5":
            self.Score.configure(text = f"Score - {Score / 4}/5")
        elif choice == "/4": 
            self.Score.configure(text = f"Score - {Score / 5}/4")
        elif choice == "Percent":
            self.Score.configure(text = f"Score - {Score * 5}%")
            
    def changemode(self):
        if self.modebutton.cget("image") == LightMode:
            self.modebutton.configure(image=DarkMode, fg_color = Basic_Background_ColorD)
            ctk.set_appearance_mode("dark")
            self.banner.configure(image = DarkLogo)
            self.configure(fg_color = Basic_Background_ColorD)
        
        else:
            self.modebutton.configure(image=LightMode, fg_color = Basic_Background_ColorL)
            ctk.set_appearance_mode("light")
            self.banner.configure(image = Logo)
            self.configure(fg_color = Basic_Background_ColorL)

    def exitWindow(self):
        self.destroy()
        self.master.destroy()

    def clear(self):
        self.text_area.delete("1.0", "end")

    def analyze_sentiment(self, comment: str) -> pd.DataFrame:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        lemma = WordNetLemmatizer()
        stop_words = stopwords.words('english')
        global sentiment_pos
        def text_prep(x: str) -> list:
            corp = str(x).lower() 
            corp = re.sub('[^a-zA-Z]+',' ', corp).strip() 
            tokens = word_tokenize(corp)
            words = [t for t in tokens if t not in stop_words]
            lemmatize = [lemma.lemmatize(w) for w in words]
            return lemmatize

        # Create a DataFrame with the provided comment
        df = pd.DataFrame({'review_content': [comment]})

        # Apply text preprocessing to the comment
        preprocess_tag = [text_prep(i) for i in df['review_content']]
        df["preprocess_txt"] = preprocess_tag
        df['total_len'] = df['preprocess_txt'].map(lambda x: len(x))

        # Load positive and negative word lists
        file = open('negative_word_list.txt', 'r')
        neg_words = file.read().split()
        file = open('positive_word_list.txt', 'r')
        pos_words = file.read().split()

        # Count positive and negative words in the comment
        num_pos = df['preprocess_txt'].map(lambda x: len([i for i in x if i in pos_words]))
        df['pos_count'] = num_pos
        num_neg = df['preprocess_txt'].map(lambda x: len([i for i in x if i in neg_words]))
        df['neg_count'] = num_neg

        # Compute sentiment score
        df['sentiment'] = round((df['pos_count'] - df['neg_count']) / df['total_len'], 2)
        df['predicted_sentiment'] = sentiment_pos

        return df
    def train_thread(self):
        # Load the data
        """
        rotten_tomatoes_reviews_adjusted = pd.read_csv('Rotten_Tomato_Sentiment.csv')

        # Drop rows with missing values in relevant columns
        rotten_tomatoes_reviews_adjusted = rotten_tomatoes_reviews_adjusted.dropna(subset=['sentiment', 'neg_count', 'pos_count', 'total_len'])

        # Split the data
        train, test = train_test_split(rotten_tomatoes_reviews_adjusted, test_size=0.25, train_size=0.75, random_state=42)

        # Define features and labels
        feature_cols = ['sentiment', 'pos_count']
        train_y = train['review_score']
        test_y = test['review_score']

        # Define additional features
        train_x_other = train[feature_cols].reset_index(drop=True)
        test_x_other = test[feature_cols].reset_index(drop=True)

        # Define and train the model (using Linear Regression for regression task)
        linear_reg = LinearRegression()
        linear_reg.fit(train_x_other, train_y)

        # Evaluate the model
        predictions = linear_reg.predict(test_x_other)
        mse = mean_squared_error(test_y, predictions)
        print(f"Mean Squared Error: {mse}")
        """
        feature_cols = ['sentiment', 'neg_count', 'pos_count', 'total_len', 'predicted_sentiment']
        with open('model_sentiment_values' , 'rb') as f:
            linear_reg = pickle.load(f)
        # Make predictions for a sample review
        review = self.text_area.get("1.0", "end - 1 chars")
        sample_df = self.analyze_sentiment(review)[feature_cols]
        prediction = linear_reg.predict(sample_df)
        self.sentimentscore.configure( text = f"Sentiment Score - {prediction}")
        self.length.configure( text = f"Length - {sample_df['total_len'][0]}")
        self.num_negative.configure( text = f"Num of Negative Words -  {sample_df['neg_count'][0]}")
        self.num_positive.configure( text = f"Num of Positive Words - {sample_df['pos_count'][0]}")
        global Score
        Score = prediction[0]
        self.Score.configure( text = f"Score - {prediction[0]}")

    def estimate_thread(self):
        text = 	self.text_area.get("1.0", "end - 1 chars")
        
        semanlysis_dataset = pd.read_csv("IMDB Dataset.csv")
        # Split the data
        train, test = train_test_split(semanlysis_dataset, test_size=0.25, train_size=.75, random_state=42)

        with open('model_tfidf' , 'rb') as f:
            tfidf = pickle.load(f)

        # Define features and labels
        test_x, test_y = test['review'], test['sentiment']
        test_x_vector = tfidf.transform(test_x)

        # load saved model
        with open('model_log_reg' , 'rb') as f:
            semanlysis_Pos_Neg = pickle.load(f)

        
        # check prediction
        prediction = semanlysis_Pos_Neg.predict(tfidf.transform([text])) # similar
        accuracy = semanlysis_Pos_Neg.score(test_x_vector, test_y)
        #self.sentimentaccuracy.configure(text = f"Sentiment Accuracy - {accuracy}")
        self.sentimentvar.configure(text =  f"Sentiment - {prediction}")
        self.sentimentaccuracy.configure(text = f"Accuracy - {accuracy}")
        
        global sentiment_pos
        if prediction == "positive":
            sentiment_pos = 1.0
            print("Phrase is positive")
        elif prediction == "negative":
            sentiment_pos = 0.0
            print("Phrase is negative")
        else: 
            print(f"Didn't work. value is {prediction} with value {sentiment_pos}")

    def estimate(self):
        self.estimate_thread()
        self.train_thread()
        """
        estimate_thread = threading.Thread(target=self.estimate_thread)
        train_thread = threading.Thread(target=self.train_thread)

        estimate_thread.start()
        train_thread.start()

        estimate_thread.join()
        train_thread.join()
        """
ctk.set_appearance_mode("system")
root = StartWindow()
root.mainloop()    





