# Agent review: agent_eval_dev_n50_v2.jsonl


29 of 50 questions shown (citation precision < 1 or declined)


## Q3. by how much did nus outperform abus?

citation precision: 0.0 | gold reached the model: 1/1 | declined: False

**Reference answers:** Average success rate is higher by 2.6 percent points.

**Agent answer:** NUS outperformed ABUS by 3.4 percentage points in success rate (93.4 % vs 90.0 %)[E1].

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

**Agent answer:** They leverage prior knowledge in the form of class‑indicative words (e.g., “NBA”, “player”, “basketball” for sports or sentiment‑bearing words such as “terrible”, “perfect”) that are manually provided as labeled features to guide learning[E2][E3]

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

**Agent answer:** Yes; the subject‑verb agreement, reflexive anaphora, and negative‑polarity‑item tasks were evaluated in earlier studies that used uni‑directional LSTM models [E2].

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

**Agent answer:** Current entity‑linking systems often produce linked entities that are too ambiguous and overly coarse, so many of them are irrelevant or incorrect for summarization; for example, a large share of extracted entities have Wikipedia disambiguation pages (71 % of the top‑100 and 53.6 % of random samples) and a substantial portion are tagged as irrelevant or wrong (30 % irrelevant, 26.9 % incorrect)[E2]. This ambiguity and coarseness limits the usefulness of the extracted entities in generating accurate summaries[E1].

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

citation precision: 0.25 | gold reached the model: 1/1 | declined: False

**Reference answers:** Distributions of Followers, Friends and URLs are significantly different between the set of tweets containing fake news and those non containing them, but for Favourites, Mentions, Media, Retweets and Hashtags they are not significantly different

**Agent answer:** The study identified 1 785 855 tweets, of which 1 327 went viral (≥ 1 000 retweets) and 136 of those viral tweets were labeled as containing fake news [E1]. Compared with viral tweets not containing fake news, the fake‑news tweets showed statistically significant distribution differences in several meta‑data dimensions: a higher proportion of unverified accounts, a larger friends‑to‑followers ratio, fewer mentions per tweet, fewer media elements, more URLs, fewer favourites, and a larger number of hashtags, while the number of followers also differed significantly [E2][E4]. All of these differences were confirmed by Kolmogorov‑Smirnov tests on the respective attribute distributions [E6].

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

❌ `1712.05999::007` (Data and Methodology)
> In the following section, we provide graphical descriptions of the distribution of each of the identified attributes for the two sets of tweets (those labelled as containing fake news and those labelled as not containing them). Where appropriate, we normalized and/or took logarithms of the data for better representation. To gain a better understanding of the significance of those differences, we use the Kolmogorov-Smirnov test with the null hypothesis that both distributions are equal.

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

**Agent answer:** Back‑translation works because it turns abundant monolingual target text into artificial parallel data that is easier for the NMT to learn from: the pseudo‑source sentences are on average shorter, syntactically simpler, and have more monotonic word alignments than natural source sentences, which facilitates translation learning[E2][E5].

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

**Agent answer:** The K‑Nearest Neighbors classifier achieved the highest scores among the models tested, but overall the classifiers showed only modest performance and did not perform particularly well [E1][E2].

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

## Q22. What are the source and target domains?

citation precision: 0.0 | gold reached the model: 2/2 | declined: False

**Reference answers:** Book, electronics, beauty, music, IMDB, Yelp, cell phone, baby, DVDs, kitchen || we use set 1 of the source domain as the only source with sentiment label information during training, and we evaluate the trained model on set 1 of the target domain; Book (BK), Electronics (E), Beauty (BT), and Music (M)

**Agent answer:** The source domain is the existing domain that provides sufficient labeled data, while the target domain is the new domain for which only few or no labeled data are available[E1].

**Cited passages:**

❌ `1809.00530::001` (Introduction)
> In practice, it is often difficult and costly to annotate sufficient training data for diverse application domains on-the-fly. We may have sufficient labeled data in an existing domain (called the source domain), but very few or no labeled data in a new domain (called the target domain). This issue has motivated research on cross-domain sentiment classification, where knowledge in the source domain is transferred to the target domain in order to alleviate the required labeling effort.
One key ch…

**Gold passages:**

(in evidence) `1809.00530::010` (Datasets and Experimental Settings)
> Existing benchmark datasets such as the Amazon benchmark [ref] typically remove reviews with neutral labels in both domains. This is problematic as the label information of the target domain is not accessible in an unsupervised domain adaptation setting. Furthermore, removing neutral instances may bias the dataset favorably for max-margin-based algorithms like ours, since the resulting dataset has all uncertain labels removed, leaving only high confidence examples. Therefore, we construct new da…

(in evidence) `1809.00530::019` (Results on Amazon Benchmark)
> Most previous works [ref] , [ref] , [ref] , [ref] , [ref] carried out experiments on the Amazon benchmark released by Blitzer et al. ( [ref] ). The dataset contains 4 different domains: Book (B), DVDs (D), Electronics (E), and Kitchen (K). Following their experimental settings, we consider the binary classification task to predict whether a review is positive or negative on the target domain. Each domain consists of 1000 positive and 1000 negative reviews respectively. We also allow 4000 unlabel…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q23. What were the previously reported results?

citation precision: 0.0 | gold reached the model: 1/1 | declined: False

**Reference answers:** Figures FIGREF1 and FIGREF1 contain a summary of the papers addressing speculation detection and scope resolution

**Agent answer:** The paper notes that prior work had already achieved state‑of‑the‑art performance on three of the four negation‑scope datasets, and that those existing systems were surpassed by the new approach [E1][E2].

**Cited passages:**

❌ `2001.02885::002` (Introduction)
> Inspired by the most recent approach of applying BERT to negation detection and scope resolution ([ref]), we take this approach one step further by performing a comparative analysis of three popular transformer-based architectures: BERT ([ref]), XLNet ([ref]) and RoBERTa ([ref]), applied to speculation detection and scope resolution. We evaluate the performance of each model across all datasets via the single dataset training approach, and report all scores including inter-dataset scores (i.e. t…

❌ `2001.02885::006` (Negation Cue Detection and Scope Resolution)
> We use a default train-validation-test split of 70-15-15 for each dataset, and use all 4 datasets (BF, BA, SFU and Sherlock). The results for BERT are taken from [ref]. The results for XLNet and RoBERTa are averaged across 5 runs for statistical significance. Figure contains results for negation cue detection and scope resolution. We report state-of-the-art results on negation scope resolution on BF, BA and SFU datasets. Contrary to popular opinion, we observe that XLNet is better than RoBERTa f…

**Gold passages:**

(in evidence) `2001.02885::001` (Introduction)
> The task of speculation detection and scope resolution is critical in distinguishing factual information from speculative information. This has multiple use-cases, like systems that determine the veracity of information, and those that involve requirement analysis. This task is particularly important to the biomedical domain, where patient reports and medical articles often use this feature of natural language. This task is commonly broken down into two subtasks: the first subtask, speculation c…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q24. Why masking words in the decoder is helpful?

citation precision: 0.5 | gold reached the model: 1/1 | declined: False

**Reference answers:** ecause this process is similar to the cloze task in BERT's pre-train process, therefore by using the ability of the contextual language model the decoder can generate more fluent and natural sequences.

**Agent answer:** Masking each word in the draft lets the refine decoder treat the task as a cloze‑style prediction, which matches BERT’s pre‑training objective and gives the decoder a full bidirectional context (the other summary words and the source document) when predicting the masked token; this yields more fluent and natural sequences[E1]. Additionally, this masking addresses the limitation of left‑context‑only decoders by providing the missing future context during refinement[E4].

**Cited passages:**

✅ `1902.09243::009` (Summary Refine Process)
> The main reason to introduce the refine process is to enhance the decoder using BERT's contextualized representations, so we do not modify the encoder and reuse it during this process.
On the decoder side, we propose a new word-level refine decoder. The refine decoder receives a generated summary draft as input, and outputs a refined summary. It first masks each word in the summary draft one by one, then feeds the draft to BERT to generate context vectors. Finally it predicts a refined summary w…

❌ `1902.09243::001` (Introduction)
> Text summarization generates summaries from input documents while keeping salient information. It is an important task and can be applied to several real-world applications. Many methods have been proposed to solve the text summarization problem [ref] , [ref] , [ref] , [ref] . There are two main text summarization techniques: extractive and abstractive. Extractive summarization generates summary by selecting salient sentences or phrases from the source text, while abstractive methods paraphrase …

**Gold passages:**

(in evidence) `1902.09243::009` (Summary Refine Process)
> The main reason to introduce the refine process is to enhance the decoder using BERT's contextualized representations, so we do not modify the encoder and reuse it during this process.
On the decoder side, we propose a new word-level refine decoder. The refine decoder receives a generated summary draft as input, and outputs a refined summary. It first masks each word in the summary draft one by one, then feeds the draft to BERT to generate context vectors. Finally it predicts a refined summary w…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q26. What does an individual model consist of?

citation precision: 0.0 | gold reached the model: 1/1 | declined: False

**Reference answers:** Bayesian model of garg2012unsupervised as our base monolingual model

**Agent answer:** An individual model is a Bayesian monolingual model that is built separately for each language [E1].

**Cited passages:**

❌ `1603.01514::001` (Introduction)
> Semantic Role Labeling (SRL) has emerged as an important task in Natural Language Processing (NLP) due to its applicability in information extraction, question answering, and other NLP tasks. SRL is the problem of finding predicate-argument structure in a sentence, as illustrated below:
 [math] 
Here, the predicate WRITE has two arguments: `Mike' as A0 or the writer, and `a book' as A1 or the thing written. The labels A0 and A1 correspond to the PropBank annotations [ref] .
As the need for SRL a…

**Gold passages:**

(in evidence) `1603.01514::003` (Monolingual Model)
> We use the Bayesian model of garg2012unsupervised as our base monolingual model. The semantic roles are predicate-specific. To model the role ordering and repetition preferences, the role inventory for each predicate is divided into Primary and Secondary roles as follows:
For example, the complete role sequence in a frame could be: [math] [math] , [math] , [math] , [math] , [math] , [math] , [math] , [math] [math] . The ordering is defined as the sequence of PRs, [math] [math] , [math] , [math] …

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q28. How does enforcing agreement between parse trees work across different languages?

citation precision: 0.5 | gold reached the model: 1/1 | declined: False

**Reference answers:** we can expect a dependency edge between i and i' in the English parse tree to correspond to an edge between j and j' in the Hindi parse tree; dual decomposition inference algorithm tries to bring the parse trees in the two languages through its constraints

**Agent answer:** Enforcing agreement is done by a dual‑decomposition inference that iteratively adjusts the English and Hindi parse trees so that aligned words induce matching dependency edges: an English edge (i,i′) aligned to Hindi words (j,j′) should correspond to a Hindi edge or a path between j and j′, and vice‑versa, with the algorithm alternating between fixing one tree and projecting the best compatible tree for the other until convergence[E1]. The iterative coordinate‑descent procedure calls a Project sub‑algorithm that, given a fixed source tree and word alignments, selects the target‑language tree that maximizes the summed scores of projected paths while satisfying these agreement constraints[E2].

**Cited passages:**

✅ `1603.08594::004` (Our Approach)
> We propose a Dual Decomposition based inference algorithm to look at the problem of PP-attachment disambiguation. Dual decomposition, or more generally, Lagrangian Relaxation, is a classical method for combinatorial optimization and has been applied to several inference problems in NLP [ref] . We train two separate parser models for English and Hindi each, using the MSTParser, and make use of these models in the inferencing step. The input to the algorithm is a parallel English-Hindi sentence pa…

❌ `1603.08594::006` (Dual Decomposition based Algorithm)
> We use an iterative Coordinate Descent algorithm (Algorithm 1) which calls the Project Algorithm until convergence. The trees [math] and [math] are initialized by the previously trained parser models for the respective languages.
Coordinate Descent Algorithm [1] Initialize [math] and [math] from the MSTParser models for [math] to [math] [math] [math] if [math] or [math] break else [math] [math] end for
For [math] iterations, the project function returns a parse tree for English which maximizes t…

**Gold passages:**

(in evidence) `1603.08594::004` (Our Approach)
> We propose a Dual Decomposition based inference algorithm to look at the problem of PP-attachment disambiguation. Dual decomposition, or more generally, Lagrangian Relaxation, is a classical method for combinatorial optimization and has been applied to several inference problems in NLP [ref] . We train two separate parser models for English and Hindi each, using the MSTParser, and make use of these models in the inferencing step. The input to the algorithm is a parallel English-Hindi sentence pa…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q30. what datasets did they use?

citation precision: 0.0 | gold reached the model: 0/1 | declined: False

**Reference answers:** Dataset of total 3500 questions from the Internet and other sources such as books of general knowledge questions, history, etc. || 3500 questions collected from the internet and books.

**Agent answer:** They used a Bengali question dataset consisting of 3 150 questions for training and 350 questions for validation, and a separate stop‑words dataset for the stop‑word elimination setup [E4][E1].

**Cited passages:**

❌ `1911.03059::006` (Proposed Methodology)
> We use different types of classifiers for QA type classification. We separate our methodology into two sections similar to [ref] - one is training section and another is validation section shown in Figure.
We use 10 fold cross validation where we have 3150 and 350 questions in our training set and validation set respectively. During training, after selecting the possible class labels, the system extracts the features of the questions and creates a model by passing through a classifier algorithm …

❌ `1911.03059::011` (Implementation of the System ::: Feature Extraction ::: Stop Words)
> We use two setups (as done in [ref]) for our system. In the first setup, we eliminate the stop words from the text using another dataset containing only stop words. At second step, we work without eliminating the stop words from the text which gives better result than the first setup.

**Gold passages:**

(NOT retrieved) `1911.03059::007` (Question Collection and Categories)
> Though Bengali is the seventh most spoken language in terms of number of native speakers [ref], there is no standard corpus of questions available [ref]. We have collected total 3500 questions from the Internet and other sources such as books of general knowledge questions, history etc. The corpus contains the questions and the classes each question belongs to.
The set of question categories is known as question taxonomy [ref]. We have used two layer taxonomy which was proposed by Xin Li, Dan Ro…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q31. How many electrodes were used on the subject in EEG sessions?

citation precision: - | gold reached the model: 1/1 | declined: True

**Reference answers:** 1913 signals

**Agent answer:** The provided evidence does not answer this question.

**Cited passages:**

(none)

**Gold passages:**

(in evidence) `1904.04358::010` (Training and hyperparameter selection)
> We performed two sets of experiments with the single-trial EEG data. In PHASE-ONE, our goals was to identify the best architectures and hyperparameters for our networks with a reasonable number of runs. For PHASE-ONE, we randomly shuffled and divided the data (1913 signals from 14 individuals) into train (80%), development (10%) and test sets (10%). In PHASE-TWO, in order to perform a fair comparison with the previous methods reported on the same dataset, we perform a leave-one-subject out cross…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q33. What models are used for painting embedding and what for language style transfer?

citation precision: 0.5 | gold reached the model: 1/4 | declined: False

**Reference answers:** generating a poem from images we use an existing actor-critic architecture; various types of sequence to sequence models

**Agent answer:** The painting is embedded with a CNN‑RNN based image‑to‑poem network, and the language‑style transfer is performed by a sequence‑to‑sequence model that uses global attention. [E1][E3]

**Cited passages:**

❌ `1910.03634::012` (Results ::: Conclusions and Future Work)
> In conclusion, combining two pipelines with an intermediate representation works well in practice. We observe that a CNN-RNN based image-to-poem net combined with a seq2seq model with parallel text corpus for text style transfer synthesizes Shakespeare-style prose for a given painting. For the seq2seq model used, we observe that it performs better in practice using global attention as compared with local attention. We make our models and code publicly available [ref]. In future work we would lik…

✅ `1910.03634::006` (Methods ::: Shakespearizing Poetic Captions ::: Seq2Seq with Attention)
> We use a sequence-to-sequence model which consists of a single layer unidrectional LSTM encoder and a single layer LSTM decoder and pre-trained retrofitted word embeddings shared between source and target sentences. We experimented with two different types of attention: global attention [ref], in which the model makes use of the output from the encoder and decoder for the current time step only, and Bahdanau attention [ref], where computing attention requires the output of the decoder from the p…

**Gold passages:**

(NOT retrieved) `1910.03634::004` (Methods ::: Image To Poem Actor-Critic Model)
> For generating a poem from images we use an existing actor-critic architecture [ref]. This involves 3 parallel CNNs: an object CNN, sentiment CNN, and scene CNN, for feature extraction. These features are combined with a skip-thought model which provides poetic clues, which are then fed into a sequence-to-sequence model trained by policy gradient with 2 discriminator networks for rewards. This as a whole forms a pipeline that takes in an image and outputs a poem as shown on the top left of Figur…

(NOT retrieved) `1910.03634::005` (Methods ::: Shakespearizing Poetic Captions)
> For Shakespearizing modern English texts, we experimented with various types of sequence to sequence models. Since the size of the parallel translation data available is small, we leverage a dictionary providing a mapping between Shakespearean words and modern English words to retrofit pre-trained word embeddings. Incorporating this extra information improves the translation task. The large number of shared word types between the source and target sentences indicates that sharing the representat…

(in evidence) `1910.03634::006` (Methods ::: Shakespearizing Poetic Captions ::: Seq2Seq with Attention)
> We use a sequence-to-sequence model which consists of a single layer unidrectional LSTM encoder and a single layer LSTM decoder and pre-trained retrofitted word embeddings shared between source and target sentences. We experimented with two different types of attention: global attention [ref], in which the model makes use of the output from the encoder and decoder for the current time step only, and Bahdanau attention [ref], where computing attention requires the output of the decoder from the p…

(NOT retrieved) `1910.03634::007` (Methods ::: Shakespearizing Poetic Captions ::: Seq2Seq with a Pointer Network)
> Since a pair of corresponding Shakespeare and modern English sentences have significant vocabulary overlap we extend the sequence-to-sequence model mentioned above using pointer networks [ref] that provide location based attention and have been used to enable copying of tokens directly from the input. Moreover, there are lot of proper nouns and rare words which might not be predicted by a vanilla sequence to sequence model.

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q34. How do they control for annotation artificats?

citation precision: 0.5 | gold reached the model: 2/2 | declined: False

**Reference answers:**  we use several of the MCQA baseline models first introduced in BIBREF0 || Choice-Only model, which is a variant of the well-known hypothesis-only baseline; Choice-to-choice model, tries to single out a given answer choice relative to other choices; Question-to-choice model, in contrast, uses the contextual representations for each question and individual choice and an attention model Att model to get a score

**Agent answer:** They detect and guard against annotation artifacts by training and evaluating partial‑input baseline models (e.g., Choice‑Only and Choice‑to‑Choice) that score answers without seeing the question, and by using these baselines to check whether systematic biases were introduced during data creation; when high scores appear they apply filtering steps to remove biased distractors before finalizing the probes[E1][E3]

**Cited passages:**

✅ `1912.13337::016` (Probing Methodology and Modeling ::: Task Definition and Modeling ::: Baselines and Sanity Checks.)
> When creating synthetic datasets, it is important to ensure that systematic biases, or annotation artifacts [ref], are not introduced into the resulting probes and that the target datasets are sufficiently challenging (or good, in the sense of [ref]). To test for this, we use several of the MCQA baseline models first introduced in [ref], which take inspiration from the LSTM-based models used in [ref] for NLI and various partial-input baselines based on these models.
Following the notation from […

❌ `1912.13337::022` (Results and Findings ::: Are our Probes Sufficiently Challenging?)
> As shown in Table, most of our partial-input baselines (i.e., Choice-Only and Choice-to-Choice models) failed to perform well on our dataset probes across a wide range of models, showing that such probes are generally immune from biases relating to how distractors were generated. As already discussed in Section, however, initial versions of the DictionaryQA dataset had unforeseen biases partly related to whether distractors were sampled from entries without example sentences, which resulted in h…

**Gold passages:**

(in evidence) `1912.13337::016` (Probing Methodology and Modeling ::: Task Definition and Modeling ::: Baselines and Sanity Checks.)
> When creating synthetic datasets, it is important to ensure that systematic biases, or annotation artifacts [ref], are not introduced into the resulting probes and that the target datasets are sufficiently challenging (or good, in the sense of [ref]). To test for this, we use several of the MCQA baseline models first introduced in [ref], which take inspiration from the LSTM-based models used in [ref] for NLI and various partial-input baselines based on these models.
Following the notation from […

(in evidence) `1912.13337::017` (Probing Methodology and Modeling ::: Task Definition and Modeling ::: Baselines and Sanity Checks.)
> A Question-to-choice model, in contrast, uses the contextual representations for each question and individual choice and an attention model Att model to get a score $\alpha ^{(j)}_{q,i} = \textsc {Att}(r^{(j)}_{q},r^{(j)}_{c_{i}}) \in \mathbb {R}$ as above. Here we also experiment with using ESIM [ref] to generate the contextual representations $r$, as well as a simpler VecSimilarity model that measures the average vector similarity between question and answer tokens: $\alpha ^{(j)}_{q,i} = \tex…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q36. what are the previous state of the art?

citation precision: 0.67 | gold reached the model: 2/3 | declined: False

**Reference answers:** Farasa; RDI || Farasa BIBREF31; MADAMIRA BIBREF29; RDI (Rashwan et al., 2015); MIT (Belinkow and Glass, 2015); Microsoft ATKS BIBREF28

**Agent answer:** The prior state‑of‑the‑art systems for Arabic diacritization were Farasa, MADAMIRA, RDI (Rashwan et al., 2015), MIT (Belinkow and Glass, 2015) and Microsoft ATKS, which achieved word error rates around 12 %–12.2 % for MSA and CE error rates of 10.7 %–14.4 % [E1][E5][E7].

**Cited passages:**

✅ `2002.01207::014` (Experiments and Results ::: Core Word ::: Results and Error analysis ::: MSA Results:)
> For MSA, though the CHAR+PRIOR feature led to worse results than using CHAR alone, the results show that combining all the features achieved the best results. Moreover, post correction improved results overall. We compare our results to five other systems, namely Farasa [ref], MADAMIRA [ref], RDI (Rashwan et al., 2015), MIT (Belinkow and Glass, 2015), and Microsoft ATKS [ref]. Table compares our system with others in the aforementioned systems. As the results show, our results beat the current s…

✅ `2002.01207::002` (Introduction)
> saw whom. Similarly, in the sub-sentence “kAn Alm&tmr AltAsE” (كان المؤتمر التاسع>), if the last word, is a predicate of the verb “kAn”, then the sentence would mean “this conference was the ninth” and would receive a fatHa (a) as a case ending. Conversely, if it was an adjective to the “conference”, then the sentence would mean “the ninth conference was ...” and would receive a damma (u) as a case ending. Thus, a consideration of context is required for proper disambiguation. Due to the inter-w…

❌ `2002.01207::005` (Background)
> [ref] used a hybrid approach that utilizes the output of Alkhalil morphological Analyzer [ref] to generate all possible out of context diacritizations of a word. Then, an HMM guesses the correct diacritized form. Similarly, Microsoft Arabic Toolkit Services (ATKS) diacritizer [ref] uses a rule-based morphological analyzer that produces possible analyses and an HMM in conjunction with rules to guess the most likely analysis. They report WER of 11.4% and 4.4% with and without CE. MADAMIRA [ref] us…

**Gold passages:**

(NOT retrieved) `2002.01207::001` (Introduction)
> Modern Standard Arabic (MSA) and Classical Arabic (CA) have two types of vowels, namely long vowels, which are explicitly written, and short vowels, aka diacritics, which are typically omitted in writing but are reintroduced by readers to properly pronounce words. Since diacritics disambiguate the sense of the words in context and their syntactic roles in sentences, automatic diacritic recovery is essential for applications such as text-to-speech and educational tools for language learners, who …

(in evidence) `2002.01207::002` (Introduction)
> saw whom. Similarly, in the sub-sentence “kAn Alm&tmr AltAsE” (كان المؤتمر التاسع>), if the last word, is a predicate of the verb “kAn”, then the sentence would mean “this conference was the ninth” and would receive a fatHa (a) as a case ending. Conversely, if it was an adjective to the “conference”, then the sentence would mean “the ninth conference was ...” and would receive a damma (u) as a case ending. Thus, a consideration of context is required for proper disambiguation. Due to the inter-w…

(in evidence) `2002.01207::014` (Experiments and Results ::: Core Word ::: Results and Error analysis ::: MSA Results:)
> For MSA, though the CHAR+PRIOR feature led to worse results than using CHAR alone, the results show that combining all the features achieved the best results. Moreover, post correction improved results overall. We compare our results to five other systems, namely Farasa [ref], MADAMIRA [ref], RDI (Rashwan et al., 2015), MIT (Belinkow and Glass, 2015), and Microsoft ATKS [ref]. Table compares our system with others in the aforementioned systems. As the results show, our results beat the current s…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q37. How was annotation performed?

citation precision: 0.67 | gold reached the model: 2/2 | declined: False

**Reference answers:** Experienced medical doctors used a linguistic annotation tool to annotate entities. || WebAnno

**Agent answer:** Annotation was carried out manually by medical doctors using the web‑based tool WebAnno, who could work on either a pre‑annotated version (with suggested entity spans) or a blank version of each text, preferring the latter; the annotators followed detailed guidelines to label case entities, conditions, factors, findings and modifiers and their relations[E1]. The annotation focused on the presentation sections of case reports (including titles) and covered all mentions of the targeted entities within those sections[E2]. Four doctors initially piloted the guidelines on two texts, after which two annotators completed the full corpus of 53 reports, achieving a Cohen’s κ of 0.68[E3].

**Cited passages:**

✅ `2003.13032::005` (A Corpus of Medical Case Reports with Medical Entity Annotation ::: Annotation Tools and Format)
> The annotation was performed using WebAnno [ref], a web-based tool for linguistic annotation. The annotators could choose between a pre-annotated version or a blank version of each text. The pre-annotated versions contained suggested entity spans based on string matches from lists of conditions and findings synonym lists. Their quality varied widely throughout the corpus. The blank version was preferred by the annotators. We distribute the corpus in BioC JSON format. BioC was chosen as it allows…

❌ `2003.13032::002` (A Corpus of Medical Case Reports with Medical Entity Annotation ::: Annotation tasks)
> Case reports are standardized in the CARE guidelines [ref]. They represent a detailed description of the symptoms, signs, diagnosis, treatment, and follow-up of an individual patient. We focus on documents freely available through PubMed Central (PMC). The presentation of the patient's case can usually be found in a dedicated section or the abstract. We perform a manual annotation of all mentions of case entities, conditions, findings, factors and modifiers. The scope of our manual annotation is…

✅ `2003.13032::004` (A Corpus of Medical Case Reports with Medical Entity Annotation ::: Annotators)
> We asked medical doctors experienced in extracting knowledge related to medical entities from texts to annotate the entities described above. Initially, we asked four annotators to test our guidelines on two texts. Subsequently, identified issues were discussed and resolved. Following this pilot annotation phase, we asked two different annotators to annotate two case reports according to our guidelines. The same annotators annotated an overall collection of 53 case reports.
Inter-annotator agree…

**Gold passages:**

(in evidence) `2003.13032::004` (A Corpus of Medical Case Reports with Medical Entity Annotation ::: Annotators)
> We asked medical doctors experienced in extracting knowledge related to medical entities from texts to annotate the entities described above. Initially, we asked four annotators to test our guidelines on two texts. Subsequently, identified issues were discussed and resolved. Following this pilot annotation phase, we asked two different annotators to annotate two case reports according to our guidelines. The same annotators annotated an overall collection of 53 case reports.
Inter-annotator agree…

(in evidence) `2003.13032::005` (A Corpus of Medical Case Reports with Medical Entity Annotation ::: Annotation Tools and Format)
> The annotation was performed using WebAnno [ref], a web-based tool for linguistic annotation. The annotators could choose between a pre-annotated version or a blank version of each text. The pre-annotated versions contained suggested entity spans based on string matches from lists of conditions and findings synonym lists. Their quality varied widely throughout the corpus. The blank version was preferred by the annotators. We distribute the corpus in BioC JSON format. BioC was chosen as it allows…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q39. What are the state of the art models?

citation precision: - | gold reached the model: 2/2 | declined: True

**Reference answers:** BIBREF9 ; BIBREF8  || BIBREF9 , BIBREF8

**Agent answer:** The provided evidence does not answer this question.

**Cited passages:**

(none)

**Gold passages:**

(in evidence) `1610.08815::018` (Results on Dataset 1)
> As shown in Table , for every feature CNN-SVM outperforms the performance of the CNN. Following [ref] , we have carried out a 5-fold cross-validation on this dataset. The baseline features ( Section ) perform best among other features. Among all the pre-trained models, the sentiment model (F1-score: 87.00%) achieves better performance in comparison with the other two pre-trained models. Interestingly, when we merge the baseline features with the features extracted by the pre-trained deep NLP mod…

(in evidence) `1610.08815::021` (Testing Generalizability of the Models and Discussions)
> To test the generalization capability of the proposed approach, we perform training on Dataset 1 and test on Dataset 3. The F1-score drops down dramatically to 33.05%. In order to understand this finding, we visualize each dataset using PCA (Figure ). It depicts that, although Dataset 1 is mostly linearly separable, Dataset 3 is not. A linear kernel that performs well on Dataset 1 fails to provide good performance on Dataset 3. If we use RBF kernel, it overfits the data and produces worse result…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q40. How are similarities and differences between the texts from violent and non-violent religious groups analyzed?

citation precision: 0.5 | gold reached the model: 3/3 | declined: False

**Reference answers:** By using topic modeling and unsupervised emotion detection on ISIS materials and articles from Catholic women forum || A comparison of common words; We aggregated the score for each word and normalized each article by emotions. To better compare the result, we added a baseline of 100 random articles from a Reuters news dataset as a non-religious general resource

**Agent answer:** The study compares the two corpora by first normalizing word‑frequency counts and then applying topic modeling with non‑negative matrix factorization (NMF) to extract the most coherent topics in each set of texts; overlapping topics such as motherhood, marriage/divorce, and spousal relations are identified as similarities, while other topics (e.g., early‑Islam themes in ISIS and distinct Catholic themes) are noted as differences[E1][E6].

**Cited passages:**

❌ `1912.03804::017` (Conclusion and Future Work)
> In this paper, we have applied natural language processing methods to ISIS propaganda materials in an attempt to understand these materials using available technologies. We also compared these texts with a non-violent religious groups' (both focusing on women related articles) to examine possible similarities or differences in their approaches. To compare the contents, we used word frequency and topic modeling with NMF. Also, our results showed that NMF outperforms LDA due to the niche domain an…

✅ `1912.03804::015` (Results ::: Content Analysis)
> After pre-processing the text, both corpora were analyzed for word frequencies. These word frequencies have been normalized by the number of words in each corpus. Figure shows the most common words in each of these corpora.
A comparison of common words suggests that those related to marital relationships ( husband, wife, etc.) appear in both corpora, but the religious theme of ISIS material appears to be stronger. A stronger comparison can be made using topic modeling techniques to discover main…

**Gold passages:**

(in evidence) `1912.03804::001` (Introduction)
> Since its rise in 2013, the Islamic State of Iraq and Syria (ISIS) has utilized the Internet to spread its ideology, radicalize individuals, and recruit them to their cause. In comparison to other Islamic extremist groups, ISIS' use of technology was more sophisticated, voluminous, and targeted. For example, during ISIS' advance toward Mosul, ISIS related accounts tweeted some 40,000 tweets in one day [ref].However, this heavy engagement forced social media platforms to institute policies to pre…

(in evidence) `1912.03804::015` (Results ::: Content Analysis)
> After pre-processing the text, both corpora were analyzed for word frequencies. These word frequencies have been normalized by the number of words in each corpus. Figure shows the most common words in each of these corpora.
A comparison of common words suggests that those related to marital relationships ( husband, wife, etc.) appear in both corpora, but the religious theme of ISIS material appears to be stronger. A stronger comparison can be made using topic modeling techniques to discover main…

(in evidence) `1912.03804::016` (Results ::: Emotion Analysis)
> We rely on Depechemood dictionaries to analyze emotions in both corpora. These dictionaries are freely available and come in multiple arrangements. We used a version that includes words with their part of speech (POS) tags. Only words that exist in the Depechemood dictionary with the same POS tag are considered for our analysis. We aggregated the score for each word and normalized each article by emotions. To better compare the result, we added a baseline of 100 random articles from a Reuters ne…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q41. What is the definition of offensive language?

citation precision: 0.0 | gold reached the model: 1/1 | declined: False

**Reference answers:**  Most prior work focuses on a different aspect of offensive language such as abusive language BIBREF0 , BIBREF1 , (cyber-)aggression BIBREF2 , (cyber-)bullying BIBREF3 , BIBREF4 , toxic comments INLINEFORM0 , hate speech BIBREF5 , BIBREF6 , BIBREF7 , BIBREF8 , BIBREF9 , BIBREF10 , and offensive language BIBREF11 . Prior work has focused on these aspects of offensive language in Twitter BIBREF3 , BIBREF7 , BIBREF8 , BIBREF11 , Wikipedia comments, and Facebook posts BIBREF2 .

**Agent answer:** Offensive language (OFF) is defined as any post that contains non‑acceptable language (profanity) or a targeted offense, which may be veiled or direct, and includes insults, threats, and posts containing profane or swear words[E1]

**Cited passages:**

❌ `1902.09666::004` (Level A: Offensive language Detection)
> Level A discriminates between offensive (OFF) and non-offensive (NOT) tweets.
Not Offensive (NOT): Posts that do not contain offense or profanity;
Offensive (OFF): We label a post as offensive if it contains any form of non-acceptable language (profanity) or a targeted offense, which can be veiled or direct. This category includes insults, threats, and posts containing profane language or swear words.

**Gold passages:**

(in evidence) `1902.09666::001` (Introduction)
> Offensive content has become pervasive in social media and a reason of concern for government organizations, online communities, and social media platforms. One of the most common strategies to tackle the problem is to train systems capable of recognizing offensive content, which then can be deleted or set aside for human moderation. In the last few years, there have been several studies published on the application of computational methods to deal with this problem. Most prior work focuses on a…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q44. What metrics are used for evaluation?

citation precision: 0.67 | gold reached the model: 3/3 | declined: False

**Reference answers:** Byte-Pair Encoding perplexity  (BPE PPL),
BLEU-1,
BLEU-4,
ROUGE-L,
percentage of distinct unigram (D-1),
percentage of distinct bigrams(D-2),
user matching accuracy(UMA),
Mean Reciprocal Rank(MRR)
Pairwise preference over baseline(PP) || BLEU-1/4 and ROUGE-L; likelihood of generated recipes using identical input specifications but conditioned on ten different user profiles; user matching accuracy (UMA); Mean Reciprocal Rank (MRR); neural scoring model from BIBREF33 to measure recipe-level coherence ||  Distinct-1/2, UMA = User Matching Accuracy, MRR
= Mean Reciprocal Rank, PP = Pairwise preference over baseline (evaluated for 310 recipe pairs per model)

**Agent answer:** The paper evaluates generation quality with BPE perplexity, BLEU‑1/4, ROUGE‑L, and distinct‑1/2 (percentage of distinct unigrams and bigrams) [E1]; personalization is measured with user‑matching accuracy (UMA) and mean reciprocal rank (MRR) [E2]; coherence is assessed with a recipe‑level coherence score (range 0–2) and a step‑entailment score [E2][E5]; and human judgments are reported as the proportion of pairwise preferences for personalized models (63 % of comparisons) and the percentage of users finding personalized recipes more coherent (60 %) [E5].

**Cited passages:**

✅ `1909.00105::006` (Experiments and Results)
> For training and evaluation, we provide our model with the first 3-5 ingredients listed in each recipe. We decode recipe text via top-$k$ sampling [ref], finding $k=3$ to produce satisfactory results. We use a hidden size $d_h=256$ for both the encoder and decoder. Embedding dimensions for vocabulary, ingredient, recipe, techniques, and caloric level are 300, 10, 50, 50, and 5 (respectively). For prior recipe attention, we set $k=20$, the 80th %-ile for the number of user interactions. We use th…

✅ `1909.00105::007` (Experiments and Results)
> Personalization: To measure personalization, we evaluate how closely the generated text corresponds to a particular user profile. We compute the likelihood of generated recipes using identical input specifications but conditioned on ten different user profiles—one `gold' user who consumed the original recipe, and nine randomly generated user profiles. Following [ref], we expect the highest likelihood for the recipe conditioned on the gold user. We measure user matching accuracy (UMA)—the proport…

❌ `1909.00105::008` (Experiments and Results)
> Recipe Step Entailment: Local coherence is also crucial to a user following a recipe: it is crucial that subsequent steps are logically consistent with prior ones. We model local coherence as an entailment task: predicting the likelihood that a recipe step follows the preceding. We sample several consecutive (positive) and non-consecutive (negative) pairs of steps from each recipe. We train a BERT [ref] model to predict the entailment score of a pair of steps separated by a [SEP] token, using th…

**Gold passages:**

(in evidence) `1909.00105::005` (Recipe Dataset: Food.com)
> We collect a novel dataset of 230K+ recipe texts and 1M+ user interactions (reviews) over 18 years (2000-2018) from Food.com. Here, we restrict to recipes with at least 3 steps, and at least 4 and no more than 20 ingredients. We discard users with fewer than 4 reviews, giving 180K+ recipes and 700K+ reviews, with splits as in tab:recipeixnstats.
Our model must learn to generate from a diverse recipe space: in our training data, the average recipe length is 117 tokens with a maximum of 256. There…

(in evidence) `1909.00105::006` (Experiments and Results)
> For training and evaluation, we provide our model with the first 3-5 ingredients listed in each recipe. We decode recipe text via top-$k$ sampling [ref], finding $k=3$ to produce satisfactory results. We use a hidden size $d_h=256$ for both the encoder and decoder. Embedding dimensions for vocabulary, ingredient, recipe, techniques, and caloric level are 300, 10, 50, 50, and 5 (respectively). For prior recipe attention, we set $k=20$, the 80th %-ile for the number of user interactions. We use th…

(in evidence) `1909.00105::007` (Experiments and Results)
> Personalization: To measure personalization, we evaluate how closely the generated text corresponds to a particular user profile. We compute the likelihood of generated recipes using identical input specifications but conditioned on ten different user profiles—one `gold' user who consumed the original recipe, and nine randomly generated user profiles. Following [ref], we expect the highest likelihood for the recipe conditioned on the gold user. We measure user matching accuracy (UMA)—the proport…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q45. What are the baselines?

citation precision: - | gold reached the model: 0/1 | declined: True

**Reference answers:** Answer with content missing: (Experimental Setup missing subsections)
To be selected: We compared REFRESH against a baseline which simply selects the first m leading sentences from each document (LEAD) and two neural models similar to ours (see left block in Figure 1), both trained with cross-entropy loss.
Answer: LEAD

**Agent answer:** The provided evidence does not answer this question.

**Cited passages:**

(none)

**Gold passages:**

(NOT retrieved) `1802.08636::009` (Experimental Setup)
> In this section we present our experimental setup for assessing the performance of our model which we call Refresh as a shorthand for REinFoRcement Learning-based Extractive Summarization. We describe our datasets, discuss implementation details, our evaluation protocol, and the systems used for comparison.

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q47. did they compare with other extractive summarization methods?

citation precision: 0.0 | gold reached the model: 1/2 | declined: False

**Reference answers:** No

**Agent answer:** No; the paper does not compare its approach with other extractive summarization methods[E1][E2].

**Cited passages:**

❌ `1810.12085::006` (Upper Bound on Summarization)
> For each of the 55,177 discharge summary reports in the MIMIC database, we calculated what fraction of the CUIs in the discharge summary could be found in the remaining notes for the patient's current admission ( [math] ) and in their entire longitudinal medical record ( [math] ). Table shows the CUI recall averaged across all discharge summaries by both subject_id and hadm_id. The low recall suggests that clinicians may incorporate information in the discharge note that had not been previously …

❌ `1810.12085::005` (Upper Bound on Summarization)
> Extractive summarization of discharge summaries relies on the assumption that the information in the discharge summary is documented elsewhere in the rest of the patient's notes. However, sometimes clinicians will document information in the discharge summary that may have been discussed throughout the hospital visit, but was never documented in the EHR. Thus, our first aim was to determine the upper bound of extractive summarization.
For each patient, we compared the text of the discharge summa…

**Gold passages:**

(NOT retrieved) `1810.12085::008` (Labeling History of Present Illness Notes)
> Contextual Representation Layer: The contextual representation layer takes as input the sequence of word embeddings, [math] , and outputs an embedding of the contextual representation for each word in the HPI note. The word embeddings are fed into a bi-directional LSTM, which outputs [math] , a concatenation of the hidden states of the two LSTMs for each word.
Label Scoring Layer: At this point, each word [math] is associated with a hidden representation of the word, [math] . In the label scorin…

(in evidence) `1810.12085::009` (Labeling History of Present Illness Notes)
> We evaluated the impact of character embeddings, the choice of pretrained w2v embeddings, and the addition of learned word embeddings on model performance on the dev set. We report performance of the best performing model on the test set.
Table compares dev set performance of the model using various pretrained word embeddings, with and without character embeddings, and with pretrained versus learned word embeddings. The first row in each section is the performance of the model architecture descr…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q49. What kind of issues (that are not on the forefront of computational text analysis) do they tackle?

citation precision: 0.0 | gold reached the model: 1/2 | declined: False

**Reference answers:** identifying the questions we wish to explore; Can text analysis provide a new perspective on a “big question” that has been attracting interest for years?; How can we explain what we observe?; hope to connect to multiple disciplines

**Agent answer:** They highlight “thorny” problems such as how to operationalize contested social concepts (e.g., hate speech), how cultural and social context and biases affect data and model validity, and the difficulty of ensuring interpretability and conceptual soundness of computational methods—issues that are rarely the focus of standard computational‑text‑analysis work[E1][E2]

**Cited passages:**

❌ `1907.01468::002` (Introduction)
> These are just a small sample of the many opportunities and challenges faced in computational analyses of textual data. New possibilities and frustrating obstacles emerge at every stage of research, from identification of the research question to interpretation of the results. In this article, we take the reader through a typical research process that involves measuring social or cultural concepts using computational methods, discussing both the opportunities and complications that often arise. …

❌ `1907.01468::001` (Introduction)
> In June 2015, the operators of the online discussion site Reddit banned several communities under new anti-harassment rules. [ref] used this opportunity to combine rich online data with computational methods to study a current question: Does eliminating these “echo chambers” diminish the amount of hate speech overall? Exciting opportunities like these, at the intersection of “thick” cultural and societal questions on the one hand, and the computational analysis of rich textual data on larger-tha…

**Gold passages:**

(in evidence) `1907.01468::003` (Research questions)
> We typically start by identifying the questions we wish to explore. Can text analysis provide a new perspective on a “big question” that has been attracting interest for years? Or can we raise new questions that have only recently emerged, for example about social media? For social scientists working in computational analysis, the questions are often grounded in theory, asking: How can we explain what we observe? These questions are also influenced by the availability and accessibility of data s…

(NOT retrieved) `1907.01468::004` (Research questions)
> Domain experts and fellow researchers can provide feedback on questions and help with dynamically revising them. For example, they may say “we already think we know that”, “that's too naïve”, “that doesn't reflect social reality” (negative); “two major camps in the field would give different answers to that question” (neutral); “we tried to look at that back in the 1960s, but we didn't have the technology” (positive); and “that sounds like something that people who made that archive would love”,…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---

## Q50. What dataset is used?

citation precision: 0.0 | gold reached the model: 1/1 | declined: False

**Reference answers:** English WIKIBIO, French WIKIBIO , German WIKIBIO  || WikiBio dataset;  introduce two new biography datasets, one in French and one in German

**Agent answer:** The experiments use the WikiBio biography dataset (≈700 K infobox‑description pairs) as the primary dataset[E3].

**Cited passages:**

❌ `1804.07789::003` (Introduction)
> Next, we observe that while rendering the output, once the model pays attention to a field (say, occupation) it needs to stay on this field for a few timesteps (till all the occupations are produced in the output). We refer to this as the stay on behavior. Further, we note that once the tokens of a field are referred to, they are usually not referred to later. For example, once all the occupations have been listed in the output we will never visit the occupation field again because there is noth…

**Gold passages:**

(in evidence) `1804.07789::011` (Datasets)
> We use the WikiBio dataset introduced by lebret2016neural. It consists of [math] biography articles from English Wikipedia. A biography article corresponds to a person (sportsman, politician, historical figure, actor, etc.). Each Wikipedia article has an accompanying infobox which serves as the structured input and the task is to generate the first sentence of the article (which typically is a one-line description of the person). We used the same train, valid and test sets which were made public…

**Your verdict** (edit): cited passages support the answer? yes / partly / no


---