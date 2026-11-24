# References cited in the Milestone 1 slides

Every entry below is cited somewhere in `slides/milestone1.html` / `slides/HateXAI_Milestone1.pptx` (short keys in the
slides are the author-year form). All entries were checked against a primary source; the URL column is the record that was
read. BibTeX keys match `paper/references.bib`, which now contains all of them.

One citation was **corrected during preparation**: an earlier draft attributed the use–mention / abuse-vs-mention
distinction to “Dalvi et al.”, which could not be verified. It is now attributed to Gligorić et al. (2024), which states the
distinction explicitly.

## Detection and datasets

| Slide key | Reference | Source |
| --- | --- | --- |
| Davidson 2017 | Davidson, T., Warmsley, D., Macy, M., Weber, I. *Automated Hate Speech Detection and the Problem of Offensive Language.* ICWSM 2017. `davidson2017hate` | https://ojs.aaai.org/index.php/ICWSM/article/view/14955 |
| Zampieri 2019 | Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., Kumar, R. *Predicting the Type and Target of Offensive Posts in Social Media.* NAACL-HLT 2019, 1415–1420. `zampieri2019olid` | https://aclanthology.org/N19-1144/ |
| Mandl 2019 | Mandl, T., Modha, S., Majumder, P., Patel, D., Dave, M., Mandlia, C., Patel, A. *Overview of the HASOC Track at FIRE 2019.* FIRE 2019. `mandl2019hasoc` | https://dl.acm.org/doi/10.1145/3368567.3368584 |
| Mathew 2021 | Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., Mukherjee, A. *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.* AAAI 2021. `mathew2021hatexplain` | https://ojs.aaai.org/index.php/AAAI/article/view/17745 |
| ElSherief 2021 | ElSherief, M., Ziems, C., Muchlinski, D., Anupindi, V., Seybolt, J., De Choudhury, M., Yang, D. *Latent Hatred: A Benchmark for Understanding Implicit Hate Speech.* EMNLP 2021, 345–363. `elsherief2021latent` | https://aclanthology.org/2021.emnlp-main.29/ |
| Kiela 2020 | Kiela, D., Firooz, H., Mohan, A., Goswami, V., Singh, A., Ringshia, P., Testuggine, D. *The Hateful Memes Challenge.* NeurIPS 2020. `kiela2020hatefulmemes` | https://proceedings.neurips.cc/paper/2020/hash/1b84c4cee2b8b3d823b30e2d604b1878-Abstract.html |
| Gomez 2020 | Gomez, R., Gibert, J., Gomez, L., Karatzas, D. *Exploring Hate Speech Detection in Multimodal Publications* (MMHS150K). WACV 2020, 1459–1467. `gomez2020exploring` | https://arxiv.org/abs/1910.03814 |

## Models, explainability and alignment

| Slide key | Reference | Source |
| --- | --- | --- |
| Conneau 2020 | Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., Stoyanov, V. *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R). ACL 2020. `conneau2020xlmr` | https://aclanthology.org/2020.acl-main.747/ |
| Caselli 2021 | Caselli, T., Basile, V., Mitrović, J., Granitzer, M. *HateBERT: Retraining BERT for Abusive Language Detection in English.* WOAH 2021. `caselli2021hatebert` | https://aclanthology.org/2021.woah-1.3/ |
| Lees 2022 | Lees, A., Tran, V. Q., Tay, Y., Sorensen, J., Gupta, J., Metzler, D., Vasserman, L. *A New Generation of Perspective API: Efficient Multilingual Character-level Transformers.* KDD 2022, 3197–3207. `lees2022perspective` | https://research.google/pubs/a-new-generation-of-perspective-api-efficient-multilingual-character-level-transformers/ |
| Ribeiro 2016 | Ribeiro, M. T., Singh, S., Guestrin, C. *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (LIME). KDD 2016. `ribeiro2016lime` | https://dl.acm.org/doi/10.1145/2939672.2939778 |
| Lundberg 2017 | Lundberg, S. M., Lee, S.-I. *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS 2017. `lundberg2017shap` | https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html |
| Jacovi 2020 | Jacovi, A., Goldberg, Y. *Towards Faithfully Interpretable NLP Systems.* ACL 2020. `jacovi2020faithfulness` | https://aclanthology.org/2020.acl-main.386/ |
| Ouyang 2022 | Ouyang, L., Wu, J., Jiang, X., et al. *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT). NeurIPS 2022. `ouyang2022instructgpt` | https://arxiv.org/abs/2203.02155 |
| Rafailov 2023 | Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., Finn, C. *Direct Preference Optimization.* NeurIPS 2023. `rafailov2023dpo` | https://arxiv.org/abs/2305.18290 |

## Evaluation, uncertainty and responsible moderation

| Slide key | Reference | Source |
| --- | --- | --- |
| Guo 2017 | Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. *On Calibration of Modern Neural Networks.* ICML 2017, PMLR 70, 1321–1330. `guo2017calibration` | https://proceedings.mlr.press/v70/guo17a.html |
| Madras 2018 | Madras, D., Pitassi, T., Zemel, R. *Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer.* NeurIPS 2018. `madras2018predict` | https://papers.nips.cc/paper_files/paper/2018/hash/09d37c08f7b129e96277388757530c72-Abstract.html |
| Borkan 2019 | Borkan, D., Dixon, L., Sorensen, J., Thain, N., Vasserman, L. *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.* WWW Companion 2019, 491–500. `borkan2019nuanced` | https://arxiv.org/abs/1903.04561 |
| Sap 2019 | Sap, M., Card, D., Gabriel, S., Choi, Y., Smith, N. A. *The Risk of Racial Bias in Hate Speech Detection.* ACL 2019. `sap2019racial` | https://aclanthology.org/P19-1163/ |
| Wiegand 2019 | Wiegand, M., Ruppenhofer, J., Kleinbauer, T. *Detection of Abusive Language: the Problem of Biased Datasets.* NAACL-HLT 2019, 602–608. `wiegand2019biased` | https://aclanthology.org/N19-1060/ |
| Vidgen 2020 | Vidgen, B., Derczynski, L. *Directions in Abusive Language Training Data, a Systematic Review: Garbage In, Garbage Out.* PLOS ONE 15(12): e0243300, 2020. `vidgen2020directions` | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243300 |
| MacAvaney 2019 | MacAvaney, S., Yao, H.-R., Yang, E., Russell, K., Goharian, N., Frieder, O. *Hate Speech Detection: Challenges and Solutions.* PLOS ONE 14(8): e0221152, 2019. `macavaney2019hatespeech` | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221152 |
| Fortuna 2018 | Fortuna, P., Nunes, S. *A Survey on Automatic Detection of Hate Speech in Text.* ACM Computing Surveys 51(4), 2018. `fortuna2018survey` | https://dl.acm.org/doi/10.1145/3232676 |
| Gligorić 2024 | Gligorić, K., Cheng, M., Zheng, L., Durmus, E., Jurafsky, D. *NLP Systems That Can't Tell Use from Mention Censor Counterspeech, but Teaching the Distinction Helps.* NAACL-HLT 2024, 5942–5959. `gligoric2024use` | https://aclanthology.org/2024.naacl-long.331/ |

## Link verification

Every link above was fetched and checked against the named paper. Two kinds of caveat:

- The ACM Digital Library blocks automated fetches; the three ACM DOIs (Mandl 2019, Ribeiro 2016, Fortuna 2018) were
  instead confirmed through their Crossref metadata records, which match the titles and author lists.
- One link was **wrong on first writing**: the NeurIPS URL for Kiela et al. resolved to a different paper
  (“Reinforcement Learning with Feedback Graphs”). It has been replaced with the correct proceedings page. This is a
  reminder to re-check any link that looks plausible rather than assumed.

## Notes

- `paper/references.bib` contains one entry that no slide or paper section cites: `conneau2019xnli` (XNLI). It is left in
  place and unchanged; it was already flagged as unused in `PROJECT_AUDIT.md` §4.5.
- Runnable experimental results shown on slides 5–6 and 8 come from `outputs/eval/*.json`, regenerated with
  `python experiments/run_experiments.py --config configs/default.yaml` (see `outputs/eval/run_manifest.json`). They are
  **not** citations and should never be presented without their sample sizes.
- Slide 2's two examples are records `pr_q001` and `pr_q004` from `data/probes/hard_cases.jsonl`, with the measured
  probabilities recorded in `outputs/eval/probes.json`.
