import explainer
import bert_model_wrapper
import re
import os
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from official.nlp.bert import tokenization


def cleaning_function_bert(text):
    text = re.sub("@\S+", " ", text) # Remove Mentions
    text = re.sub("https*\S+", " ", text) # Remove URL
    text = re.sub("#\S+", " ", text) # Remove Hastags
    text = re.sub('&lt;/?[a-z]+&gt;', '', text) # Remove special Charaters
    text = re.sub('#39', ' ', text) # Remove special Charaters
    text = re.sub('<.*?>', '', text) # Remove html
    text = re.sub(' +', ' ', text) # Merge multiple blank spaces
    text = text.replace("<br>", "")
    text = text.replace("</br>", "")
    return text


def load_model(model_directory):
    """Loads the fine-tuned model from directory.

    Args:
        model_directory (str): PATH string of the fine-tuned model's directory on local disk
    """
    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(model_directory, "assets", "vocab.txt"), do_lower_case=True)

    print(type(tokenizer))

    model = keras.models.load_model(model_directory)

    max_seq_length = 256

    extractor = tf.keras.Model(inputs=model.inputs,
                               outputs=[model.get_layer("bert_layer").output])

    model.summary()

    return model, tokenizer, max_seq_length, extractor


def load_imdb_dataset(train_slice, test_slice, as_supervised=False):
    (train_data, test_data), info = tfds.load('imdb_reviews',
                                              split=['train[{}]'.format(train_slice), 'test[{}]'.format(test_slice)],
                                              as_supervised=as_supervised,
                                              with_info=True
                                              )

    return train_data, test_data


def load_dataset_from_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    return df


if __name__ == "__main__":

    model_dir = "saved_models/fine_tuned/20201117_bert_model_imdb_reviews_exp_0"

    model, tokenizer, max_seq_length, extractor = load_model(model_dir)

    model_wrapper = bert_model_wrapper.BertModelWrapper(model, extractor, tokenizer, label_list=[0, 1], max_seq_len=max_seq_length, clean_function=cleaning_function_bert)

    texts =["This film is very awful. I have never seen such a bad movie.", "I really love this film. It is probably one of my favourite movie of ever.",
            "The book is very nice. The film instead is not as good as the book. I expected more for this movie.",
            "i really this film . it is probably one of my movie of ever ."]

    tokens = tokenizer.tokenize(texts[0])
    print("Tokens: {}".format(tokens))
    word_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("Word Ids: {}".format(word_ids))

    df = load_dataset_from_csv("saved_models/fine_tuned/20201117_bert_model_imdb_reviews_exp_0/df_test.csv")

    texts = df["text"][1000:1003].tolist()
    true_labels = df["label"][1000:1003].tolist()

    #true_labels = None
    print(texts)

    #embeddings = model_wrapper.extract_embedding(input_texts=texts, batch_size=32)

    #print(embeddings)

    """
    texts = ['''This was one of the most dishonest, meaningless, and non-peaceful of the films I have ever seen. The representation of the other, of the Israelis, was racist, backward, and unfair. For one, the song played on E.S' car radio when pulled up alongside a very right-wing Israeli driver was "I put a spell on you" by Natacha Atlas. The song's style is quite Arabic, but it was released on an Israeli compilation CD, and I have even heard it on the radio in Israel. Many Israeli songs (as well as architecture, foods, and slang) are influenced by Arabic culture, and there is no reason an Israeli Jew would be offended or angered by a nearby car playing that song. The way E.S. appears so calm and collected with his sunglasses and cool glare, via a long, still shot, is meant to force the viewer into seeing the Jew as haggard and racist, and E.S. as noble and temperate.I have traveled all over Israel, and I have never seen an IDF recruitment poster, since service is mandatory. But in the film, not only is there a recruitment poster, but it depicts a stereotypical image of an Arab terrorist and the words "want to shoot?" This is an extremely inaccurate depiction of the mentality of the majority of Israelis as well as Israeli soldiers, and such an "advertisement" wouldn't even exist on a random Israeli highway. In including it, the director aims to convince the audience that Israel is a society of anti-Arab racists hell-bent on murder.The ninja scene was gratuitous and needlessly violent. A Hollywood-style action scene involving Israeli soldiers shooting Palestinians would be just as unwelcome in an Israeli-directed film as the ninja scene should have been. But for some reason, images of an unrealistic, non-comic, and violent scenario manage to elicit applause from the audience since the director has smeared the Israeli side so much beforehand, that any shot of Israeli soldiers being killed would be welcome. The director shows absolutely no attempt at building bridges, portraying the "other" as human, or working towards peace; violence is made to be the only solution. This is furthered by scenes of exploding tanks, falling guard towers, and other random acts of destruction. One of my best friends serves in the Israeli military, and the targets in firing ranges are never Arab women dressed in black, or any other quasi-civilian on canvas. Soldiers at checkpoints are instructed not to fire at the head of an approaching Palestinian unless it is clear that their own lives are in danger; the method, according to my friend, is to provide a warning shout, fire into the air or around the area, and then if all else fails, shoot in the leg and then interrogate and hospitalize. Arbitrarily targeting a woman in the head, as shown in the film, is not the proper procedure.Besides these inaccuracies, the directing style was also poor. Repetition became repetitious, and no longer captivating. Symbols, such as the balloon with Arafat drawn on it, are forced outside any plot structure or effective integration in the setting; the balloon is Palestine penetrating and regaining Jerusalem, and it is created for no reason by E.S. The ambulance being checked for permits by Israeli soldiers followed by subsequent Israeli ambulances flying past the checkpoint is an overly-overt claim of an Israeli double standard by the director. The attempt by the director to show life in Nazareth as dreary and pointless is done with overkill; showing the routines of random people over and over again, even with a slight change each time, and emphasizing that not one member of the cast ever smiles and is minimalist in dialogue almost screams out the purpose of such scenes, the dreariness of life, without allowing much room for personal interpretation. By contrasting one "section" of the movie, daily life in Nazareth, with the second section, the checkpoint between Ramallah and Israel, the director subtly blames this dreariness on Israel, but never provides any direct evidence as to why such blame can be properly argued.I spent hours trying to figure out why music ended abruptly and began abruptly, and why many modern fashion-show-like and metal-action tracks were included in the score. I still cannot come up with an answer. I felt that the music was out of place in this film; the contrast between more silent scenes and intense scenes was actually annoying and not affecting or thought-provoking. I can understand if the director intended for the music to provide some comic aspect to certain scenes, but I found that there was nothing comic to be found in Israeli soldiers shooting at targets or fighting a ninja, or a woman having to suffer another walk through a checkpoint, albeit defiantly. In fact, I was tempted to close my ears during intense scenes, and annoyed by the lack of a score during quiet scenes. Whatever the director's intent, it provided only an audial displeasure throughout the film.This film has no legitimate political message because it provides an inaccurate and extreme representation of the other, and neglects to actually address any issues. It is a propaganda film, because the director intends various symbols, styles, and scenes to draw sympathy for the Palestinian side, while displaying the Israeli side as cruel and inhuman without exception; the vibrant atmosphere of an action-packed Hollywood scene or of intense music is displayed in every act of violence by Palestinians against Israelis, such that the almost inevitably positive and thrilled feelings the music and cinematography elicit from the audience are directed to one side. There is no thought, reflection, or deepening of the understanding of the conflict by the audience; emotions are simply pulled to one side, and kept there, in a "good vs bad" clichÃ© scenario. I believe this film lacked the depth, quality, and power of other Palestinian films, such as "Paradise Now" and "Wedding in the Galilee." ''',
             
             '''this was one of the most dishonest , meaningless , and non - peaceful of the films i have ever seen . the representation of the other , of the israelis , was racist , backward , and unfair . for one , the song played on e . s ' car radio when pulled up alongside a very right - wing israeli driver was " i put a spell on you " by natacha atlas . the song ' s style is quite arabic , but it was released on an israeli compilation cd , and i have even heard it on the radio in israel . many israeli songs ( as well as architecture , foods , and slang ) are influenced by arabic culture , and there is no reason an israeli jew would be offended or angered by a nearby car playing that song . the way e . s . appears so calm and collected with his sunglasses and cool glare , via a long , still shot , is meant to force the viewer into seeing the jew as haggard and racist , and e . s . as noble and temperate . i have traveled all over israel , and i have never seen an idf recruitment poster , since service is mandatory . but in the film , not only is there a recruitment poster , but it depicts a stereotypical image of an arab terrorist and the words " want''',
             '''This was one of the most dishonest, meaningless, and non-peaceful of the films I have ever seen. The representation of the other, of the Israelis, was racist, backward, and unfair. For one, the song played on E.S' car radio when pulled up alongside a very right-wing Israeli driver was "I put a spell on you" by Natacha Atlas. The song's style is quite Arabic, but it was released on an Israeli compilation CD, and I have even heard it on the radio in Israel. Many Israeli songs (as well as architecture, foods, and slang) are influenced by Arabic culture, and there is no reason an Israeli Jew would be offended or angered by a nearby car playing that song. The way E.S. appears so calm and collected with his sunglasses and cool glare, via a long, still shot, is meant to force the viewer into seeing the Jew as haggard and racist, and E.S. as noble and temperate.I have traveled all over Israel, and I have never seen an IDF recruitment poster, since service is mandatory. But in the film, not only is there a recruitment poster, but it depicts a stereotypical image of an Arab terrorist and the words "want to shoot?" This is an extremely inaccurate depiction of the mentality of the majority of Israelis as well as Israeli soldiers, and such an "advertisement" wouldn't even exist on a random Israeli highway. In including it, the director aims to convince the audience that Israel is a society of anti-Arab racists hell-bent on murder.The ninja scene was gratuitous and needlessly violent. A Hollywood-style action scene involving Israeli soldiers shooting Palestinians would be just as unwelcome in an Israeli-directed film as the ninja scene should have been.  scenario manage to elicit applause from the audience since the director has smeared the Israeli side so much beforehand, that any shot of Israeli soldiers being killed would be welcome. The director shows absolutely no attempt at building bridges, portraying the "other" as human, or working towards peace; violence is made to be the only solution. This is furthered by scenes of exploding tanks, falling guard towers, and other random acts of destruction. One of my best friends serves in the Israeli military, and the targets in firing ranges are never Arab women dressed in black, or any other quasi-civilian on canvas. Soldiers at checkpoints are instructed not to fire at the head of an approaching Palestinian unless it is clear that their own lives are in danger; the method, according to my friend, is to provide a warning shout, fire into the air or around the area, and then if all else fails, shoot in the leg and then interrogate and hospitalize. Arbitrarily targeting a woman in the head, as shown in the film, is not the proper procedure.Besides these inaccuracies, the directing style was also poor. Repetition became repetitious, and no longer captivating. Symbols, such as the balloon with Arafat drawn on it, are forced outside any plot structure or effective integration in the setting; the balloon is Palestine penetrating and regaining Jerusalem, and it is created for no reason by E.S. The ambulance being checked for permits by Israeli soldiers followed by subsequent Israeli ambulances flying past the checkpoint is an overly-overt claim of an Israeli double standard by the director. The attempt by the director to show life in Nazareth as dreary and pointless is done with overkill; showing the routines of random people over and over again, even with a slight change each time, and emphasizing that not one member of the cast ever smiles and is minimalist in dialogue almost screams out the purpose of such scenes, the dreariness of life, without allowing much room for personal interpretation. By contrasting one "section" of the movie, daily life in Nazareth, with the second section, the checkpoint between Ramallah and Israel, the director subtly blames this dreariness on Israel, but never provides any direct evidence as to why such blame can be properly argued.I spent hours trying to figure out why music ended abruptly and began abruptly, and why many modern fashion-show-like and metal-action tracks were included in the score. I still cannot come up with an answer. I felt that the music was out of place in this film; the contrast between more silent scenes and intense scenes was actually annoying and not affecting or thought-provoking. I can understand if the director intended for the music to provide some comic aspect to certain scenes, but I found that there was nothing comic to be found in Israeli soldiers shooting at targets or fighting a ninja, or a woman having to suffer another walk through a checkpoint, albeit defiantly. In fact, I was tempted to close my ears during intense scenes, and annoyed by the lack of a score during quiet scenes. Whatever the director's intent, it provided only an audial displeasure throughout the film.This film has no legitimate political message because it provides an inaccurate and extreme representation of the other, and neglects to actually address any issues. It is a propaganda film, because the director intends various symbols, styles, and scenes to draw sympathy for the Palestinian side, while displaying the Israeli side as cruel and inhuman without exception; the vibrant atmosphere of an action-packed Hollywood scene or of intense music is displayed in every act of violence by Palestinians against Israelis, such that the almost inevitably positive and thrilled feelings the music and cinematography elicit from the audience are directed to one side. There is no thought, reflection, or deepening of the understanding of the conflict by the audience; emotions are simply pulled to one side, and kept there, in a "good vs bad" clichÃ© scenario. I believe this film lacked the depth, quality, and power of other Palestinian films, such as "Paradise Now" and "Wedding in the Galilee." ''',


             '''This was one of the most dishonest, meaningless, and non-peaceful of the films I have ever seen. The representation of the other, of the Israelis, was racist, backward, and unfair. For one, the song played on E.S' car radio when pulled up alongside a very right-wing Israeli driver was "I put a spell on you" by Natacha Atlas. The song's style is quite Arabic, but it was released on an Israeli compilation CD, and I have even heard it on the radio in Israel. Many Israeli songs (as well as architecture, foods, and slang) are influenced by Arabic culture, and there is no reason an Israeli Jew would be offended or angered by a nearby car playing that song. The way E.S. appears so calm and collected with his sunglasses and cool glare, via a long, still shot, is meant to force the viewer into seeing the Jew as haggard and racist, and E.S. as noble and temperate.I have traveled all over Israel, and I have never seen an IDF recruitment poster, since service is mandatory. But in the film, not only is there a recruitment poster, but it depicts a stereotypical image of an Arab terrorist and the words "want to shoot?" This is an extremely inaccurate depiction of the mentality of the l as Israeli soldiers, and such an "advertisement" wouldn't even exist on a random Israeli highway. In including it, the director aims to convince the audience that Israel is a society of anti-Arab racists hell-bent on murder.The ninja scene was gratuitous and needlessly violent. A Hollywood-style action scene involving Israeli soldiers shooting Palestinians would be just as unwelcome in an Israeli-directed film as the ninja scene should have been. But for some reason, images of an unrealistic, non-comic, and violent scenario manage to elicit applause from the audience since the director has smeared the Israeli side so much beforehand, that any shot of Israeli soldiers being killed would be welcome. The director shows absolutely no attempt at building bridges, portraying the "other" as human, or working towards peace; violence is made to be the only solution. This is furthered by scenes of exploding tanks, falling guard towers, and other random acts of destruction. One of my best friends serves in the Israeli military, and the targets in firing ranges are never Arab women dressed in black, or any other quasi-civilian on canvas. Soldiers at checkpoints are instructed not to fire at the head of an approaching Palestinian unless it is clear that their own lives are in danger; the method, according to my friend, is to provide a warning shout, fire into the air or around the area, and then if all elsnterrogate and hospitalize. Arbitrarily targeting a woman in the head, as shown in the film, is not the proper procedure.Besides these inaccuracies, the directing style was also poor. Repetition became repetitious, and no longer captivating. Symbols, such as the balloon with Arafat drawn on it, are forced outside any plot structure or effective integration in the setting; the balloon ng and regaining Jerusalem, and it is created for no reason by E.S. The ambulance being checked for permits by Israeli soldiers followed by subsequent Israeli ambulances flying past the checkpoint is an overly-overt claim of an Israeli double standard by the director. The attempt by the director to show life in Nazareth as dreary and pointless is done with overkill; showing the routines of random people over and over again, even with a slight change each ti of the cast ever smiles and is minimalist in dialogue almost screams out the purpose of such scenes, the dreariness of life, without allowing much room for personal interpretation. By contrasting one "section" of the movie, daily life in Nazareth,  and Israel, the director subtly blames this dreariness on Israel, but never provides any direct evidence as to why such blame can be properly argued.I spent hours trying to figure out why music ended abruptly and began abruptly, and why many modern fashion-show-like and metal-action tracks were included in the score. I stil music was out of place in this film; the contrast between more silent scenes and intense scenes was actually annoying and not affecting or thought-provoking. I can understand if the director inc aspect to certain scenes, but I found that there was nothing comic to be found in Israeli soldiers shooting at targets or fighting a ninja, h a checkpoint, albeit defiantly. In fact, I was tempted to close my ears during intense scenes, and annoyed by the lack of a score during quiet scenes. Whatever the director's intent, it provided only an audial displeasure throughout the film.This film has no legitimate political message because it provides an inaccurate and extreme representation of the other, and neglects to actually address any rector intends various symbols, styles, and scenes to draw sympathy for the Palestinian side, while displaying the Israeli side as cruel and inhuman without exception; the vibrant atmosphere of an action-packed Hollywood scene or of intense music is displayed in every act of violence by Palestinians against Israelis, such that the almost inevitably positive and thrilled feelings the music and cinematography elicit from the audience are directed to one side. pulled to one side, and kept there, in a "good vs bad" clichÃ© scenario. I believe this film lacked the depth, quality, and power of other Palestinian films, such as "Paradise Now" and "Wedding in the Galilee." ''',

        ]

    predictions = model_wrapper.predict(texts)

    print(predictions)
    """
    exp = explainer.LocalExplainer(model_wrapper, "20201117_bert_model_imdb_reviews_exp_0")

    exp.fit_transform(input_texts=texts,
                      classes_of_interest=[-1]*len(texts),
                      expected_labels=true_labels,
                      flag_pos=True,
                      flag_sen=True,
                      flag_mlwe=True,
                      flag_combinations=True)



