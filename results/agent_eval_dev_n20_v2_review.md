# Agent review: agent_eval_dev_n20_v2.jsonl


10 of 20 questions shown (citation precision < 1 or declined)


## Q3. by how much did nus outperform abus?

citation precision: 0.0 | gold reached the model: 1/1 | declined: False

**Reference answers:** Average success rate is higher by 2.6 percent points.

**Agent answer:** The best policy trained on the NUS achieved a 93.4 % success rate, which is 3.4 percentage points higher than the 90.0 % success rate of the best ABUS‑trained policy, and its average reward was 13.8 versus 13.3 for ABUS, a gain of 0.5 points[E1].

**Cited passages:**

❌ `1805.06966::020` (Human Evaluation)
> The results of the human evaluation are shown in Table for 250 dialogues per policy. In Table policies are marked using an ID ( [math] ) that translates to results in Tables and Table . Both policies trained with the NUS outperformed those trained on the ABUS in terms of both reward and success rate. The best performing policy trained on the NUS achieves a 93.4% success rate and 13.8 average rewards whilst the best performing policy trained with the ABUS achieves only a 90.0% success rate and 13…

**Gold passages:**

(in evidence) `1805.06966::019` (Cross-Model Evaluation)
> Table shows the results of the cross-model evaluation after 4000 training dialogues. The policies trained with the NUS achieved an average success rate (SR) of 94.0% and of 96.6% when tested on the ABUS and the NUS, respectively. By comparison, the policies trained with the ABUS achieved average SRs of 99.5% and 45.5% respectively. Thus, training with the NUS leads to policies that can perform well on both USs, which is not the case for training with the ABUS. Furthermore, the best SRs when test…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q4. What background knowledge do they leverage?

citation precision: 0.5 | gold reached the model: 2/2 | declined: False

**Reference answers:** labeled features || labelled features, which are words whose presence strongly indicates a specific class or topic

**Agent answer:** They leverage prior knowledge in the form of class‑specific indicator words (e.g., “NBA”, “player”, “basketball” for sports or sentiment‑bearing words such as “terrible”, “perfect”) that are manually provided as labeled features to guide learning[E2][E3]

**Cited passages:**

❌ `1503.00841::001` (Introduction)
> We posses a wealth of prior knowledge about many natural language processing tasks. For example, in text categorization, we know that words such as NBA, player, and basketball are strong indicators of the sports category [ref] , and words like terrible, boring, and messing indicate a negative polarity while words like perfect, exciting, and moving suggest a positive polarity in sentiment classification.
A key problem arisen here, is how to leverage such knowledge to guide the learning process, a…

✅ `1503.00841::003` (Method)
> We address the robustness problem on top of GE-FL [ref] , a GE method which leverages labeled features as prior knowledge. A labeled feature is a strong indicator of a specific class and is manually provided to the classifier. For example, words like amazing, exciting can be labeled features for class positive in sentiment classification.

**Gold passages:**

(in evidence) `1503.00841::003` (Method)
> We address the robustness problem on top of GE-FL [ref] , a GE method which leverages labeled features as prior knowledge. A labeled feature is a strong indicator of a specific class and is manually provided to the classifier. For example, words like amazing, exciting can be labeled features for class positive in sentiment classification.

(in evidence) `1503.00841::009` (Data Preparation)
> We evaluate our methods on several commonly used datasets whose themes range from sentiment, web-page, science to medical and healthcare. We use bag-of-words feature and remove stopwords in the preprocess stage. Though we have labels of all documents, we do not use them during the learning process, instead, we use the label of features.
The movie dataset, in which the task is to classify the movie reviews as positive or negtive, is used for testing the proposed approaches with unbalanced labeled…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q5. Were any of these tasks evaluated in any previous work?

citation precision: 0.0 | gold reached the model: 3/3 | declined: False

**Reference answers:** Yes || Yes

**Agent answer:** Yes; the subject‑verb agreement, “colorless green ideas” nonce sentences, and manually constructed minimal‑pair tasks were all evaluated in earlier studies [E2].

**Cited passages:**

❌ `1901.05287::003` (Previous setups)
> All three previous work use uni-directional language-model-like models.
 [ref] start with existing sentences from wikipedia that contain a present-tense verb. They feed each sentence word by word into an LSTM, stop right before the focus verb, and ask the model to predict a binary plural/singular decision (supervised setup) or compare the probability assigned by a pre-trained language model (LM) to the plural vs singular forms of the verb (LM setup). The evaluation is then performed on sentences…

**Gold passages:**

(in evidence) `1901.05287::001` (Introduction)
> The recently introduced BERT model [ref] exhibits strong performance on several language understanding benchmarks. To what extent does it capture syntax-sensitive structures?
Recent work examines the extent to which RNN-based models capture syntax-sensitive phenomena that are traditionally taken as evidence for the existence in hierarchical structure. In particular, in [ref] we assess the ability of LSTMs to learn subject-verb agreement patterns in English, and evaluate on naturally occurring wi…

(in evidence) `1901.05287::002` (Methodology)
> I use the stimuli provided by [ref] , [ref] , [ref] , but change the experimental protocol to adapt it to the bidirectional nature of the BERT model. This requires discarding some of the stimuli, as described below. Thus, the numbers are not strictly comparable to those reported in previous work.

(in evidence) `1901.05287::006` (Results)
> Tables 1 , 2 and 3 show the results. All cases exhibit high scores—in the vast majority of the cases substantially higher than reported in previous work. As discussed above, the results are not directly comparable to previous work: the BERT models are trained on different (and larger) data, are allowed to access the suffix of the sentence in addition to its prefix, and are evaluated on somewhat different data due to discarding OOV items. Still, taken together, the high performance numbers indica…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q6. Why are current ELS's not sufficiently effective?

citation precision: 0.5 | gold reached the model: 1/1 | declined: False

**Reference answers:** Linked entities may be ambiguous or too common || linked entities extracted from ELS's have issues because of low precision rates BIBREF11 and design challenges in training datasets BIBREF12 . These issues can be summarized into two parts: ambiguity and coarseness.; the linked entities may also be too common to be considered an entity.

**Agent answer:** Current entity‑linking systems often produce linked entities that are too ambiguous and overly coarse, so many of them are irrelevant or incorrect for summarization; for example, a large share of extracted entities have disambiguation pages (71 % of the top‑100 and 53.6 % of random samples) and a substantial portion are tagged as irrelevant or wrong (30 % irrelevant, 26.9 % incorrect)[E2]. This ambiguity and coarseness limits their usefulness in generating accurate summaries[E1].

**Cited passages:**

✅ `1806.05504::005` (Possible issues)
> Despite its usefulness, linked entities extracted from ELS's have issues because of low precision rates [ref] and design challenges in training datasets [ref] . These issues can be summarized into two parts: ambiguity and coarseness.
First, the extracted entities may be ambiguous. In the example, the entity “South Korean” is ambiguous because it can refer to both the South Korean person and the South Korean language, among others. In our experimental data, we extract (1) the top 100 entities bas…

❌ `1806.05504::002` (Introduction)
> To this end, we present a method to effectively apply linked entities in sequence-to-sequence models, called Entity2Topic (E2T). E2T is a module that can be easily attached to any sequence-to-sequence based summarization model. The module encodes the entities extracted from the original text by an entity linking system (ELS), constructs a vector representing the topic of the summary to be generated, and informs the decoder about the constructed topic vector. Due to the imperfections of current E…

**Gold passages:**

(in evidence) `1806.05504::005` (Possible issues)
> Despite its usefulness, linked entities extracted from ELS's have issues because of low precision rates [ref] and design challenges in training datasets [ref] . These issues can be summarized into two parts: ambiguity and coarseness.
First, the extracted entities may be ambiguous. In the example, the entity “South Korean” is ambiguous because it can refer to both the South Korean person and the South Korean language, among others. In our experimental data, we extract (1) the top 100 entities bas…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q9. How many users do they look at?

citation precision: 0.5 | gold reached the model: 2/2 | declined: False

**Reference answers:** 22,880 users || 20,000

**Agent answer:** They examine a total of 22,880 users[E1][E2]

**Cited passages:**

✅ `1612.08205::005` (Dataset)
> We compile our industry-annotated dataset by identifying blogger profiles located in the U.S. on the profile finder on http://www.blogger.com, and scraping only those users that had the industry profile element completed.
For each of these bloggers, we retrieve all their blogs, and for each of these blogs we download the 21 most recent blog postings. We then clean these blog posts of HTML tags and tokenize them, and drop those bloggers whose cumulative textual content in their posts is less than…

❌ `1612.08205::006` (Text-based Industry Modeling)
> After collecting our dataset, we split it into three sets: a train set, a development set, and a test set. The sizes of these sets are 17,880, 2,500, and 2,500 users, respectively, with users randomly assigned to these sets. In all the experiments that follow, we evaluate our classifiers by training them on the train set, configure the parameters and measure performance on the development set, and finally report the prediction accuracy and results on the test set. Note that all the experiments a…

**Gold passages:**

(in evidence) `1612.08205::001` (Introduction)
> Over the past two decades, the emergence of social media has enabled the proliferation of traceable human behavior. The content posted by users can reflect who their friends are, what topics they are interested in, or which company they are working for. At the same time, users are listing a number of profile fields to define themselves to others. The utilization of such metadata has proven important in facilitating further developments of applications in advertising [ref] , personalization [ref]…

(in evidence) `1612.08205::005` (Dataset)
> We compile our industry-annotated dataset by identifying blogger profiles located in the U.S. on the profile finder on http://www.blogger.com, and scraping only those users that had the industry profile element completed.
For each of these bloggers, we retrieve all their blogs, and for each of these blogs we download the 21 most recent blog postings. We then clean these blog posts of HTML tags and tokenize them, and drop those bloggers whose cumulative textual content in their posts is less than…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q10. What were their distribution results?

citation precision: 0.33 | gold reached the model: 1/1 | declined: False

**Reference answers:** Distributions of Followers, Friends and URLs are significantly different between the set of tweets containing fake news and those non containing them, but for Favourites, Mentions, Media, Retweets and Hashtags they are not significantly different

**Agent answer:** The study identified 1 785 855 tweets from 848 196 users, of which 1 327 went viral (≥ 1 000 retweets) and were manually labeled; 136 of these viral tweets were classified as containing fake news while the remaining 1 191 were not [E1]. Compared with non‑fake‑news viral tweets, the fake‑news tweets showed statistically significant distribution differences in several meta‑data dimensions: a higher proportion of unverified accounts, a larger friends‑to‑followers ratio, fewer mentions per tweet, more URLs per tweet, and a distinct (significantly different) distribution of follower counts; differences in media presence, favourites, and hashtag counts were not statistically significant [E2][E4].

**Cited passages:**

✅ `1712.05999::008` (Results)
> The sample collected consisted on 1 785 855 tweets published by 848 196 different users. Within our sample, we identified 1327 tweets that went viral (retweeted more than 1000 times by the 8th of November 2016) produced by 643 users. Such small subset of viral tweets were retweeted on 290 841 occasions in the observed time-window.
The 1327 `viral' tweets were manually annotated as containing fake news or not. The annotation was carried out by a single person in order to obtain a consistent annot…

❌ `1712.05999::010` (Characterization)
> We found that 82 users within our sample were spreading fake news (i.e. they produced at least one tweet which was labelled as fake news). Out of those, 34 had verified accounts, and the rest were unverified. From the 48 unverified accounts, 6 have been suspended by Twitter at the date of writing, 3 tried to imitate legitimate accounts of others, and 4 accounts have been already deleted. Figure shows the proportion of verified accounts to unverified accounts for viral tweets (containing fake new…

❌ `1712.05999::009` (Exposure)
> Figure shows that, in contrast to other kinds of viral tweets, those containing fake news were created more recently. As such, Twitter users were exposed to fake news related to the election for a shorter period of time.
However, in terms of retweets, Figure shows no apparent difference between containing fake news or not containing them. That is confirmed by the Kolmogorov-Smirnoff test, which does not discard the hypothesis that the associated distributions are equal.
In relation to the number…

**Gold passages:**

(in evidence) `1712.05999::008` (Results)
> The sample collected consisted on 1 785 855 tweets published by 848 196 different users. Within our sample, we identified 1327 tweets that went viral (retweeted more than 1000 times by the 8th of November 2016) produced by 643 users. Such small subset of viral tweets were retweeted on 290 841 occasions in the observed time-window.
The 1327 `viral' tweets were manually annotated as containing fake news or not. The annotation was carried out by a single person in order to obtain a consistent annot…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q13. Does the dataset feature only English language data?

citation precision: - | gold reached the model: 1/1 | declined: True

**Reference answers:** Yes

**Agent answer:** The provided evidence does not answer this question.

**Cited passages:**

(none)

**Gold passages:**

(in evidence) `1808.10245::001` (Introduction)
> Abusive language refers to any type of insult, vulgarity, or profanity that debases the target; it also can be anything that causes aggravation [ref] , [ref] . Abusive language is often reframed as, but not limited to, offensive language [ref] , cyberbullying [ref] , othering language [ref] , and hate speech [ref] .
Recently, an increasing number of users have been subjected to harassment, or have witnessed offensive behaviors online [ref] . Major social media companies (i.e. Facebook, Twitter) …

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q15. what is their explanation for the effectiveness of back-translation?

citation precision: 0.5 | gold reached the model: 1/1 | declined: False

**Reference answers:** when using BT, cases where the source is shorter than the target are rarer; cases when they have the same length are more frequent; automatic word alignments between artificial sources tend to be more monotonic than when using natural sources

**Agent answer:** Back‑translation works because it turns abundant monolingual target text into artificial parallel data that NMT can train on, and the resulting pseudo‑source sentences have properties that make learning easier – they tend to be shorter, more monotonic in word alignment and syntactically simpler than natural sources, which facilitates translation performance[E2][E5]

**Cited passages:**

❌ `1903.11437::001` (Introduction)
> The new generation of Neural Machine Translation (NMT) systems is known to be extremely data hungry [ref] . Yet, most existing NMT training pipelines fail to fully take advantage of the very large volume of monolingual source and/or parallel data that is often available. Making a better use of data is particularly critical in domain adaptation scenarios, where parallel adaptation data is usually assumed to be small in comparison to out-of-domain parallel data, or to in-domain monolingual texts. …

✅ `1903.11437::008` (Properties of back-translated data)
> Comparing the natural and artificial sources of our parallel data wrt. several linguistic and distributional properties, we observe that (see Fig. - Figure ):
artificial sources are on average shorter than natural ones: when using BT, cases where the source is shorter than the target are rarer; cases when they have the same length are more frequent.
automatic word alignments between artificial sources tend to be more monotonic than when using natural sources, as measured by the average Kendall […

**Gold passages:**

(in evidence) `1903.11437::008` (Properties of back-translated data)
> Comparing the natural and artificial sources of our parallel data wrt. several linguistic and distributional properties, we observe that (see Fig. - Figure ):
artificial sources are on average shorter than natural ones: when using BT, cases where the source is shorter than the target are rarer; cases when they have the same length are more frequent.
automatic word alignments between artificial sources tend to be more monotonic than when using natural sources, as measured by the average Kendall […

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q19. What is the performance of classifiers?

citation precision: 0.5 | gold reached the model: 1/1 | declined: False

**Reference answers:** Table TABREF10;  The KNN classifier seem to perform the best across all four metrics. This is probably due to the multi-class nature of the data set;  While these classifiers did not perform particularly well, they provide a good starting point for future work on this subject || Using F1 Micro measure, the KNN classifier perform 0.6762, the RF 0.6687, SVM 0.6712 and MLP 0.6778.

**Agent answer:** The K‑Nearest Neighbors classifier achieved the highest scores among the four models (KNN, RF, SVM, MLP) on both micro‑ and macro‑averaged F1 in 4‑fold cross‑validation, although overall the classifiers “did not perform particularly well” [E1][E2].

**Cited passages:**

✅ `2002.02070::008` (Translating Car-Speak ::: Classification Experiments)
> We train a series of classifiers in order to classify car-speak. We train three classifiers on the review vectors that we prepared in Section. The classifiers we use are K Nearest Neighbors (KNN), Random Forest (RF), Support Vector Machine (SVM), and Multi-layer Perceptron (MLP) [ref].
In order to evaluate our classifiers, we perform 4-fold cross validation on a shuffled data set. Table shows the F1 micro and F1 macro scores for all the classifiers. The KNN classifier seem to perform the best ac…

❌ `2002.02070::009` (Conclusion & Future Work)
> In this paper we aim to provide an introductory understanding of car-speak and a way to automate car dealers at dealerships. We first provide a definition of “car-speak” in Section. We explore what constitutes car-speak and how to identify car-speak.
We also gather a data set of car-speak to use for exploration and training purposes. This data set id full of vehicle reviews from U.S. News [ref]. These reviews provide a reasonable set of car-speak data that we can study.
Finally, we create and te…

**Gold passages:**

(in evidence) `2002.02070::008` (Translating Car-Speak ::: Classification Experiments)
> We train a series of classifiers in order to classify car-speak. We train three classifiers on the review vectors that we prepared in Section. The classifiers we use are K Nearest Neighbors (KNN), Random Forest (RF), Support Vector Machine (SVM), and Multi-layer Perceptron (MLP) [ref].
In order to evaluate our classifiers, we perform 4-fold cross validation on a shuffled data set. Table shows the F1 micro and F1 macro scores for all the classifiers. The KNN classifier seem to perform the best ac…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q20. By how much more does PARENT correlate with human judgements in comparison to other text generation metrics?

citation precision: - | gold reached the model: 2/2 | declined: True

**Reference answers:** Best proposed metric has average correlation with human judgement of 0.913 and 0.846 compared to best compared metrics result of 0.758 and 0.829 on WikiBio and WebNLG challenge. || Their average correlation tops the best other model by 0.155 on WikiBio.

**Agent answer:** The provided evidence does not answer this question.

**Cited passages:**

(none)

**Gold passages:**

(in evidence) `1906.01081::009` (Correlation Comparison)
> We use bootstrap sampling (500 iterations) over the 1100 tables for which we collected human annotations to get an idea of how the correlation of each metric varies with the underlying data. In each iteration, we sample with replacement, tables along with their references and all the generated texts for that table. Then we compute aggregated human evaluation and metric scores for each of the models and compute the correlation between the two. We report the average correlation across all bootstra…

(in evidence) `1906.01081::013` (WebNLG Dataset)
> To check how PARENT correlates with human judgments when the references are elicited from humans (and less likely to be divergent), we check its correlation with the human ratings provided for the systems competing in the WebNLG challenge [ref] . The task is to generate text describing 1-5 RDF triples (e.g. John E Blaha, birthPlace, San Antonio), and human ratings were collected for the outputs of 9 participating systems on 223 instances. These systems include a mix of pipelined, statistical and…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---