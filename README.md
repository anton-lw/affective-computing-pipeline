# A computational pipeline for longitudinal well-being analysis from a personal text corpus

This repository contains a computational pipeline designed to transform a raw, timestamped text archive into a structured, multi-dimensional time series of psychological and emotional metrics. The process involves robust data parsing, multi-layered feature annotation, temporal aggregation, and statistical analysis to identify significant shifts in linguistic patterns over time.

The initial stage of the pipeline involves parsing a text archive where messages may span multiple lines. A parser identifies new message headers using regular expressions to correctly group content, converting the unstructured text into a structured pandas dataframe with timestamp, username, and content columns. Subsequently, each message is processed through a series of annotation functions. A baseline psycholinguistic analysis is conducted using the NRC EmoLex lexicon to calculate the frequency of words associated with eight primary emotions and two sentiment polarities. The system also quantifies validated linguistic markers for self-focus (first-person singular pronoun use) and cognitive rigidity (absolutist word frequency). Concurrently, each message is scored using the VADER sentiment analysis tool to derive a continuous compound sentiment score that accounts for grammatical rules, negation, and intensifiers. For a more granular emotional profile, a fine-tuned RoBERTa-based transformer model (SamLowe/roberta-base-go_emotions) is employed for multi-label classification, assigning a probability for each of twenty-seven distinct emotions. Finally, a large language model is utilized for a zero-shot classification task, processing messages in batches to assign labels from the five PERMA pillars of well-being.

The resulting high-dimensional dataset is then aggregated into a daily time series, calculating the mean for continuous scores and the frequency for categorical labels. Change point detection is subsequently applied to a 30-day rolling average of the sentiment time series using the Pelt algorithm from the ruptures library to automatically identify statistically significant shifts in the data's properties. The final stage involves synthesizing these data streams for interpretation through a series of visualizations, including the primary sentiment trajectory with marked change points, a normalized plot of multiple psychological markers over time, and the average sentiment score associated with each PERMA pillar.

## Getting started

To use the pipeline, Python 3.9+ is required. The user must provide a text message archive in the root directory, with each new message beginning in the format `[YYYY-MM-DD HH:MM:SS] username: message content`. All necessary Python libraries should be installed from the `requirements.txt` file. Two additional files must be manually placed in the project's root directory: the NRC Emotion Lexicon file, named `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`, and a `.env` file containing an API key in the format `GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"`.

The pipeline is executed by running the main script, `a_main_analysis.py`. On the first execution, the script performs the complete annotation process and saves the annotated dataframe to `annotated_messages.csv`. On subsequent runs, the script will detect this existing file, load it, and proceed directly to the temporal analysis and visualization steps. To force a re-annotation of the source data, the `annotated_messages.csv` file must be deleted.

## Project structure

The project is organized into several modules. The main entry point that orchestrates the pipeline is `a_main_analysis.py`. The `b_data_loader.py` script handles the parsing of the input text file. All modules for annotating the data are contained within `c_feature_annotator.py`. The script `d_temporal_analyzer.py` manages the aggregation of data into time series and performs the change point detection. Finally, `e_visualizer.py` contains all plotting functions.

## Theoretical foundations and citations

The use of the NRC Emotion Lexicon for tracking word-emotion associations is based on the work of Mohammad & Turney (2013). The measurement of validated mental health markers is informed by research on self-focus, or "I-talk," as a linguistic marker for distress (Tausczik & Pennebaker, 2010; Tackman et al., 2019) and the measurement of absolutist words as a marker for cognitive rigidity (Al-Mosaiwi & Johnstone, 2018). The VADER model is used for rule-based sentiment analysis attuned to informal text as described by Hutto & Gilbert (2014). For fine-grained emotion classification, a transformer model is employed, a technique validated on datasets such as the GoEmotions dataset (Demszky et al., 2020). The zero-shot classification of text into a theoretical well-being framework is based on the PERMA model (Seligman, 2011). Finally, the temporal analysis uses the `ruptures` library for offline change point detection, which implements algorithms such as the Pelt search algorithm (Truong, Oudre, & Vayatis, 2020; Killick, Fearnhead, & Eckley, 2012).

### References

Al-Mosaiwi, M., & Johnstone, T. (2018). In an Absolute State: Elevated Use of Absolutist Words Is a Marker Specific to Anxiety, Depression, and Suicidal Ideation. *Clinical Psychological Science, 6*(4), 529–542.

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*.

Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. In *Proceedings of the International AAAI Conference on Web and Social Media, 8*(1), 216-225.

Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association, 107*(500), 1590-1598.

Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a Word-Emotion Association Lexicon. *Computational Intelligence, 29*(3), 436-465.

Seligman, M. E. P. (2011). *Flourish: A Visionary New Understanding of Happiness and Well-being*. Atria Books.

Tackman, A. M., et al. (2019). Depression, Negative Emotionality, and Self-Referential Language: A Multi-Lab, Multi-Measure, and Multi-Language-Task Research Synthesis. *Journal of Personality and Social Psychology, 116*(5), 817–834.

Tausczik, Y. R., & Pennebaker, J. W. (2010). The Psychological Meaning of Words: LIWC and Computerized Text Analysis Methods. *Journal of Language and Social Psychology, 29*(1), 24–54.

Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing, 167*, 107299.
