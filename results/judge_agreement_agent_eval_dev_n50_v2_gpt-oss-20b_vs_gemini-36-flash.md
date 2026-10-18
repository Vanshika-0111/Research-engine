# Judge agreement: openai/gpt-oss-20b (A) vs gemini-3.6-flash (B)

Agent results: `agent_eval_dev_n50_v2`; 50 questions, 45 answered.

| verdict | items | exact agreement | Cohen's kappa | lenient agreement (yes/partly vs no) |
|---|---|---|---|---|
| correct | 45 | 0.822 | 0.498 | 0.889 |
| faithful | 45 | 0.822 | 0.151 | 0.933 |
| citations | 72 | 0.819 | 0.037 | 0.889 |

| metric | A full | B full | A filtered | B filtered |
|---|---|---|---|---|
| strict_citation_precision | 0.6278 | 0.6278 | 0.6193 | 0.6193 |
| judged_citation_precision | 0.8787 | 0.9778 | 0.8759 | 0.9773 |
| faithfulness | 0.8778 | 0.9778 | 0.875 | 0.9773 |
| answer_correctness | 0.68 | 0.78 | 0.7234 | 0.8085 |
| answer_correctness_answered_only | 0.7556 | 0.8667 | 0.7727 | 0.8636 |
| judge_errors | 0 | 0 | 0 | 0 |
| n | 50 | 50 | 47 | 47 |

## Disagreements (15)

- **Q3** by how much did nus outperform abus?  
  correct A/B = no/yes, faithful = yes/yes  
  A: Answer differs from reference by 0.8 percentage points  
  B: The system correctly calculates the difference between the best performing NUS and ABUS policies (93.4% - 90.0% = 3.4 percentage points) based on E1.
- **Q5** Were any of these tasks evaluated in any previous work?  
  correct A/B = partly/yes, faithful = partly/yes, citations: E2 partly/yes  
  A: E2 only supports subject‑verb agreement  
  B: The cited passage confirms that all three tasks were evaluated in previous work using unidirectional LSTM models.
- **Q9** How many users do they look at?  
  correct A/B = yes/yes, faithful = yes/yes, citations: E2 no/yes  
  A: E2 does not support the total user count  
  B: The system answer accurately states the number of users supported by the cited passages and matching the reference answers.
- **Q10** What were their distribution results?  
  correct A/B = no/partly, faithful = no/yes, citations: E2 partly/yes, E4 no/yes  
  A: The answer overstates significance for media, hashtags, and favourites, contradicting the cited passages.  
  B: The system answer correctly identifies the statistical significance (or lack thereof) for each attribute, but conflates which dimensions were significantly different versus not significantly different into one sentence, making it slightly ambiguous compared to the reference.
- **Q15** what is their explanation for the effectiveness of back-translation?  
  correct A/B = yes/yes, faithful = partly/partly, citations: E2 partly/no  
  A: E2 does not support the learning ease claim  
  B: Citation E2 does not support the explanation details. In E5, artificial sources are NOT on average shorter than natural ones (the text states cases where source is shorter than target are rarer).
- **Q21** What is the model architecture used?  
  correct A/B = yes/yes, faithful = partly/yes, citations: E1 no/yes  
  A: Citation E1 does not support the deep‑learning part of the answer  
  B: The system answer accurately describes both architectures mentioned in the paper and reference answers.
- **Q26** What does an individual model consist of?  
  correct A/B = no/no, faithful = no/yes, citations: E1 no/yes  
  A: The cited passage discusses SRL and multilingual models, not the specific Bayesian monolingual model from garg2012unsupervised.  
  B: The system answer identifies the model as a Bayesian monolingual model from the context, but fails to mention the reference answer's specific base model (garg2012unsupervised).
- **Q27** what dataset is used in this paper?  
  correct A/B = no/yes, faithful = yes/yes  
  A: The answer cites Natura 2000, but the reference answers state the datasets are the same as BIBREF7, which is not confirmed by the cited passage.  
  B: The system answer correctly identifies the main Natura 2000 dataset mentioned in passage E2.
- **Q30** what datasets did they use?  
  correct A/B = no/yes, faithful = yes/yes  
  A: Answer does not match the reference datasets  
  B: The system answer accurately identifies the training, validation, and stop-words datasets based on the cited text.
- **Q36** what are the previous state of the art?  
  correct A/B = yes/yes, faithful = partly/yes, citations: E5 no/yes  
  A: E5 does not support the cited claim  
  B: The response accurately identifies the prior state-of-the-art systems and their performance figures as cited in the paper.
- **Q40** How are similarities and differences between the texts from violent and non-violent religi  
  correct A/B = partly/yes, faithful = yes/yes  
  A: Missing mention of emotion detection and baseline random articles  
  B: The system answer accurately describes how similarities and differences were analyzed using normalized word frequencies and NMF topic modeling.
- **Q47** did they compare with other extractive summarization methods?  
  correct A/B = yes/yes, faithful = no/yes, citations: E1 no/yes, E2 no/yes  
  A: The cited passages do not support the claim of no comparison.  
  B: The system correctly answers 'No' and the citations support that the paper focuses on upper bound analysis rather than comparing with other extractive summarization methods.
- **Q48** How do their models decide how much improtance to give to the output words?  
  correct A/B = no/no, faithful = yes/partly, citations: E1 yes/partly  
  A: The answer correctly cites the passage but does not match the reference method.  
  B: The system answer incorrectly states that the importance scores are for output words, whereas the cited passage and reference answers state they are for input words.
- **Q49** What kind of issues (that are not on the forefront of computational text analysis) do they  
  correct A/B = no/yes, faithful = partly/yes, citations: E1 partly/yes, E2 partly/yes  
  A:   
  B: The system answer accurately draws the specific thorny issues mentioned in E1 and E2, which aligns well with the broader challenges outlined in the paper.
- **Q50** What dataset is used?  
  correct A/B = yes/partly, faithful = yes/yes  
  A: All claims are supported by the cited passage.  
  B: The system answer correctly identifies the WikiBio dataset from E3, but misses mentioning the newly introduced French and German WikiBio datasets referenced in the paper.
