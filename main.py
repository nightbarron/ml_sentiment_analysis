import joblib
from pyspark.ml.classification import NaiveBayesModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import xu_ly_tieng_viet as pp
import pickle
from pyspark.ml import PipelineModel
import os
import warnings
import sys
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st


def check_sentiment(sentence, ml_model, ml_vectorizer, bd_model, pipelineModel, spark):
    # Check the sentiment of the sentence

    # ML
    tokenized_ml_sentence = ml_vectorizer.transform([sentence])
    sentiment_ml = ml_model.predict(tokenized_ml_sentence)

    # {'Good': 1, 'Bad': 0}
    label_ml = "Good" if sentiment_ml[0] == 1 else "Bad"

    # BD
    new_data = spark.createDataFrame([(sentence,)], ["Comment_Preprocessing"])
    new_data = pipelineModel.transform(new_data)

    bd_model.transform(new_data)
    sentiment_bd = bd_model.transform(new_data).select( "prediction").collect()[0][0]
    label_bd = "Good" if sentiment_bd == 1 else "Bad"

    return label_ml, label_bd

def load_model():
    # Load the machine learning model
    ml_model = joblib.load('models/ml_model/best_model.pkl')
    ml_vectorizer = pickle.load(open('models/ml_model/tfidf.pkl', 'rb'))

    # Load the big data model
    bd_model = NaiveBayesModel.load("models/bd_model/best.model")
    pipelineModel = PipelineModel.load("models/bd_model/pipeline.model")
    return ml_model, ml_vectorizer, bd_model, pipelineModel

def stat_restaurant(restaurant_id, reviews, restaurants):
    # Check id of restaurant
    if int(restaurant_id) not in restaurants['ID'].values:
        st.write("Invalid restaurant ID")
        return

    # Thong ke rating cua nha hang
    restaurant_reviews = reviews[reviews['IDRestaurant'] == int(restaurant_id)]
    if restaurant_reviews.empty:
        st.write("No review for this restaurant")
        return
    
    st.write("Restaurant's name: ```", restaurants[restaurants['ID'] == int(restaurant_id)]['Restaurant'].values[0] + "```")
    st.write("Address: ```", restaurants[restaurants['ID'] == int(restaurant_id)]['Address'].values[0] + "```")
    st.write("Active Time: ```", restaurants[restaurants['ID'] == int(restaurant_id)]['Time'].values[0] + "```")
    st.write("Lowest price:", restaurants[restaurants['ID'] == int(restaurant_id)]['Lowest_price'].values[0])
    st.write("Highest price:", restaurants[restaurants['ID'] == int(restaurant_id)]['Highest_price'].values[0])
    top_30_words_good = restaurant_reviews[restaurant_reviews['Rating_Class'] == "Good"]['Comment_Preprocessing'].str.split(expand=True).stack().value_counts().head(30)
    top_30_words_bad = restaurant_reviews[restaurant_reviews['Rating_Class'] == "Bad"]['Comment_Preprocessing'].str.split(expand=True).stack().value_counts().head(30)


    avg_rating = restaurant_reviews['Rating'].mean().round(2)
    st.write("Average rating:", avg_rating, "```/10```")

    # Remove N in top 30 words
    list_top_30_words_good = top_30_words_good.index.tolist()
    list_top_30_words_bad = top_30_words_bad.index.tolist()

    # list to string
    str_top_30_words_good = ' '.join(list_top_30_words_good)
    str_top_30_words_bad = ' '.join(list_top_30_words_bad)

    # Pos tag   
    after_postag_top_30_words_good = pp.process_postag_thesea_adj(str_top_30_words_good)
    after_postag_top_30_words_bad = pp.process_postag_thesea_adj(str_top_30_words_bad)

    # parse to list
    after_postag_top_30_words_good = str.split(after_postag_top_30_words_good)
    after_postag_top_30_words_bad = str.split(after_postag_top_30_words_bad)

    st.write("Top key words in good comments: ", ', '.join(after_postag_top_30_words_good))
    st.write("Top key words in bad comments: ", ', '.join(after_postag_top_30_words_bad))

    st.write("""- Thống kê số lượng bình luận theo Rating_Class""")
    # Thong ke so luong binh luan theo Rating_Class
    plot = sns.countplot(x='Rating_Class', data=restaurant_reviews, color="skyblue")
    # plt.show()
    st.pyplot(plot.figure)

    st.write("""- Wordcloud top 30 từ khóa trong bình luận tốt""")
    # Word cloud top 30 key work binh luan tot Comment_ADJ
    wordcloud = WordCloud(max_words=30, background_color="white").generate_from_frequencies(top_30_words_good)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Top 30 key words in good comments")
    plt.axis("off")
    # plt.show()
    st.pyplot(fig)

    st.write("""- Wordcloud top 30 từ khóa trong bình luận xấu""")
    # Word cloud top 30 key work binh luan xau Comment_ADJ
    wordcloud = WordCloud(max_words=30, background_color="white").generate_from_frequencies(top_30_words_bad)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Top 30 key words in bad comments")
    plt.axis("off")
    # plt.show()
    st.pyplot(fig)



# //////////////////////////
def main_1():

    # Initialize the spark session
    warnings.filterwarnings("ignore")
    spark = SparkSession.builder.appName("Sentiment Analysis").getOrCreate()

    # Load the model
    ml_model, ml_vectorizer, bd_model, pipelineModel = load_model()

    # Load restaurant and review data
    reviews = pd.read_csv('data/2_Reviews_preprocessing.csv')
    restaurants = pd.read_csv('data/1_Restaurants_preprocessing.csv')

    while True:
        print()
        print("==== Sentiment Analysis Project ====")
        print()
        print("1. Check the sentiment of a sentence")
        print("2. Show Restaurant's sentiment based ID")
        print("Others. Exit")

        # Get user input
        choice = input("Enter your choice: ")
        
        if choice == "1":
            sentence = input("Enter the sentence: ")

            print()
            print("=> Sentence: ", sentence)

            # Convert to UTF-8
            # sentence = sentence.encode('utf-8').decode('utf-8')

            # Preprocess the sentence
            sentence = pp.preprocess_text_all_together(sentence)

            sentiment_ml, sentiment_bd = check_sentiment(sentence, ml_model, ml_vectorizer, bd_model, pipelineModel, spark)
        
            print("=> Sentiment by ML model:", sentiment_ml)
            print("=> Sentiment by BD model:", sentiment_bd)
        elif choice == "2":
            # Get restaurant ID
            restaurant_id = input("Enter the restaurant ID: ")
            stat_restaurant(restaurant_id, reviews, restaurants)
            # Load the data
        else:
            print("Invalid choice")
            break

def gioi_thieu():
    # st.subheader("[Trang chủ](https://csc.edu.vn)")
    st.subheader("*** Giới thiệu Project")
    st.markdown("""### Mục tiêu:
- Xây dựng mô hình dự đoán cảm xúc của bình luận
- Thống kê dữ liệu bình luận của nhà hàng
            
### Công nghệ sử dụng:
- Machine Learning: XGBoost
- Big Data: Naive Bayes
        
### Quy trình xây dựng mô hình:
- Thu thập dữ liệu
- Tiền xử lý dữ liệu tiếng việt
- Xây dựng mô hình
- Đánh giá mô hình => Chọn mô hình tốt nhất
- Lưu mô hình
- Áp dụng mô hình vào thực tế
- Thống kê dữ liệu
- Xây dựng giao diện và dự đoán
        
### Tác giả:
- Huỳnh Thái Bảo
- Đặng Lê Hoàng Tuấn""")
    return


def main():

    # Initialize the spark session
    warnings.filterwarnings("ignore")
    spark = SparkSession.builder.appName("Sentiment Analysis").getOrCreate()

    # Load the model
    ml_model, ml_vectorizer, bd_model, pipelineModel = load_model()

    # Load restaurant and review data
    reviews = pd.read_csv('data/2_Reviews_preprocessing.csv')
    restaurants = pd.read_csv('data/1_Restaurants_preprocessing.csv')

    # Using menu
    st.title("Project 1: Sentiment Analysis")
    menu = ["Giới Thiệu Project", "INSIGHT tập dữ liệu", "Dự đoán mới"]
    choice = st.sidebar.selectbox('Danh mục', menu)
    if choice == menu[0]: 
        gioi_thieu()
    elif choice == menu[1]:
        st.subheader("INSIGHT tập dữ liệu")
        st.write("""- Dữ liệu: Gồm 2 tập dữ liệu về nhà hàng và bình luận khách hàng""")
        st.write("""### Dữ liệu nhà hàng:""")
        st.write(restaurants.head())
        st.write("""- Có tất cả """, len(restaurants), """nhà hàng
- Biểu đồ mối tương quan giữa nhà hàng và quận huyện""")
        plot = sns.countplot(x='District', data=restaurants, color="skyblue")
        plot.xaxis.set_tick_params(rotation=45)
        st.pyplot(plot.figure)

        st.write("""- Biểu đồ giá thấp nhất, cao nhất theo huyện""")
        district_price = restaurants.groupby('District').agg({'Lowest_price': 'mean', 'Highest_price': 'mean'}).reset_index()
        bar_width = 0.35
        index = range(len(district_price))
        fig = plt.figure(figsize= (12,8))
        plt.bar(index, district_price['Highest_price'], bar_width, label='Highest Price', color='blue')
        plt.bar([i + bar_width for i in index], district_price['Lowest_price'], bar_width, label='Lowest Price', color='red')
        plt.xlabel('District')
        plt.ylabel('Price')
        plt.title('Highest and Lowest Prices by District', loc = 'left',  fontweight = 'heavy', fontsize = 16)
        plt.xticks([i + bar_width/2 for i in index], district_price['District'])
        plt.legend()
        # plt.grid(axis = 'y', linestyle ='--')
        # sns.despine(left=False, bottom=True)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""`Comment:`  
Nhìn chung vào dữ liệu `Restaurants`, ta có một số quan sát như sau:  
- Tập dữ liệu chứa 1622 nhà hàng trong đó các nhà hàng phân bố chủ yếu ở các quận trung tâm như Quận 1, 2, 3, 4 và 5 (đều trên 195 nhà hàng)nhiều nhất là ở Quận 1 (198 nhà hàng).  
- Khung giờ mở cửa chủ yếu vào 6 giờ sáng và đóng cửa chủ yếu vào lúc 23 giờ 59 phút.  
- 75% các món ăn có giá thấp nhất rơi vào khoảng từ 1.000 - 30.000. 
- 75% các món ăn có giá cao nhất rơi vào khoảng từ 4.000 - 107.000.  

=> Đa số những nhà hàng bán online thông qua Shopee Food (ở trong dataset) tập trung ở các quận lớn, dân cư đông đúc và có mức sống tốt đồng thời có hoạt động bán hàng gần như 24/24 vì có thể để phục vụ nhu cầu ăn uống cao của người dân tại các quận trên.  
=> Giá bán ở các quận này cũng cao hơn so với các quận còn lại.  
Kết luận: Có sự phân bố nhà hàng, khung giờ bán và giá món ăn dựa theo vị trí địa lý của các quận ở Tp.HCM.""")


        st.write("""### Dữ liệu bình luận:""")
        st.write(reviews.head())

        st.write("""- Có tất cả """, len(reviews), """bình luận
- Biểu đồ tương quan ratings""")
        plt.clf()
        # Round and Group rating 1-10
        reviews['Rating_Rounded'] = reviews['Rating'].round()
        rating_group = reviews.groupby('Rating_Rounded').size().reset_index(name='Count')
        plot = sns.barplot(x='Rating_Rounded', y='Count', data=rating_group, color="skyblue")
        st.pyplot(plot.figure)

        st.write("""- Wordcloud top 30 từ khóa trong bình luận tốt""")
        top_30_words_good = reviews[reviews['Rating_Class'] == "Good"]['Comment_Preprocessing'].str.split(expand=True).stack().value_counts().head(30)
        wordcloud = WordCloud(max_words=30, background_color="white").generate_from_frequencies(top_30_words_good)
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title("Top 30 key words in good comments")
        plt.axis("off")
        st.pyplot(fig)

        st.write("""- Wordcloud top 30 từ khóa trong bình luận xấu""")
        top_30_words_bad = reviews[reviews['Rating_Class'] == "Bad"]['Comment_Preprocessing'].str.split(expand=True).stack().value_counts().head(30)
        wordcloud = WordCloud(max_words=30, background_color="white").generate_from_frequencies(top_30_words_bad)
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title("Top 30 key words in bad comments")
        plt.axis("off")
        st.pyplot(fig)

        st.markdown("""`Comment:`  
Ở dữ liệu `Reviews` chúng ta có tổng cộng 29958 User mua hàng trên tổng số 1174 nhà hàng, trong đó:  
- Lượt Rating phân bố trên thang điểm từ 0 đến 10 với 50% lượt đánh giá dưới 7.6.  
- Sau khi thực hiện việc xếp hạng đánh giá theo 2 hạng mục là 'Positive' và 'Negative' dựa trên các keyword đặc trưng trong 'Comment', ta rút ra được một số đặc tính của 2 nhóm như sau:  
    - Dựa trên các keywork đặc trưng, thì nhóm 'Positive' cho ra kết quả khá chuẩn xác khi có chỉ số Rating phân bố từ 5 đến 10.  
    - Mặc khác, dữ liệu nhóm 'Negative' cho ta sự nhìn nhận lại về độ tương quan giữa Rating và Comment trong khi có những lượt Rating rất cao nhưng Comment lại tiêu cực và ngược lại. Bằng việc sử dụng các keyword đặc trưng của từng nhóm chúng ta sẽ xác định lại các nhóm với độ chính xác cao hơn khi sử dụng dữ liệu Rating.   
- Số lượng dữ liệu 'Positive' chiếm sấp sỉ gấp 2 lần 'Negative' (có một sự mất cân bằng dữ liệu khi tập dữ liệu mang nhiều lượt đánh giá tích cực hơn)
- 50% nhà hàng chỉ có dưới 10 lượt mua.  
- Số lượng mua hàng nhiều nhất tại một nhà hàng là 100 lượt và có 140 nhà hàng đạt được điều đó.""")

    elif choice == menu[2]:  
        st.subheader("Dự đoán mới")
        st.write("""### 1. Hiển thị thông tin nhà hàng theo ID""")
        list_restaurant_id = restaurants['ID'].values
        restaurant_id = st.selectbox("Chọn ID nhà hàng", list_restaurant_id)
        if st.button("Xem"):
            stat_restaurant(restaurant_id, reviews, restaurants)

        st.write("""### 2. Dự đoán cảm xúc của bình luận""")

        # Lựa chọn nhập trực tiếp hoặc upload file
        options = ["Nhập trực tiếp", "Upload file"]
        option = st.radio("Chọn cách nhập dữ liệu", options)

        if option == options[0]:
            sentence = st.text_input("Nhập bình luận", "Nhà hàng này rất ngon")
            if st.button("Dự đoán"):
                sentiment_ml, sentiment_bd = check_sentiment(sentence, ml_model, ml_vectorizer, bd_model, pipelineModel, spark)
                # st.write("=> ML model: ```", sentiment_ml + "```")
                # st.write("=> BigData model: ```", sentiment_bd + "```")
                st.write("Predict: ```", sentiment_ml + "```")
        else:
            uploaded_file = st.file_uploader("Chọn file", type=['txt'])
            if uploaded_file is not None:
                content = uploaded_file.read().decode("utf-8")
                st.write("Content loaded successfully!!!")
                if st.button("Dự đoán"):
                    # Read the content line by line and check the sentiment
                    for sentence in content.split("\n"):
                        sentiment_ml, sentiment_bd = check_sentiment(sentence, ml_model, ml_vectorizer, bd_model, pipelineModel, spark)
                        st.write("[*] Sentence: ```", sentence + "```")
                        # st.write("=> ML model: ```", sentiment_ml + "```")
                        # st.write("=> BigData model: ```", sentiment_bd + "```")
                        st.write("Predict: ```", sentiment_ml + "```")
    else:
        st.write("Invalid choice")
        return








if __name__ == "__main__":
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.environ['PYSPARK_LOG_LEVEL'] = 'ERROR'
    main()