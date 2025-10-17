# MAP - Charting Student Math Misunderstandings

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)
[![Silver Medal](https://img.shields.io/badge/Medal-Silver%20🥈-C0C0C0?style=flat)](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/45th-place-silver-medal-a-multi-llm-ensemble-a)
[![Rank](https://img.shields.io/badge/Rank-45th-orange?style=flat)](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/45th-place-silver-medal-a-multi-llm-ensemble-a)
[![MAP@3](https://img.shields.io/badge/MAP@3-0.947-blue?style=flat)](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/45th-place-silver-medal-a-multi-llm-ensemble-a)
<img width="1525" height="911" alt="image" src="https://github.com/user-attachments/assets/3596ef03-d464-4eb3-9d46-a4f75bee5d72" />

## 🎯 Competition Overview

The MAP (Misconception Annotation Project) competition challenged participants to develop NLP models that predict students' mathematical misconceptions from their open-ended explanations. The goal was to help teachers efficiently identify and address incorrect thinking patterns that hinder student learning.

### Problem Statement

Students are often asked to explain their mathematical reasoning. These explanations provide rich insight into student thinking and often reveal underlying **misconceptions** (systematic incorrect ways of thinking).

**Example:** Students often think 0.355 is larger than 0.8 because they incorrectly apply their knowledge of whole numbers to decimals, reasoning that 355 is greater than 8.

Tagging students' explanations as containing potential misconceptions is valuable for diagnostic feedback but is **time-consuming and challenging to scale**. This competition aimed to create ML-driven solutions that can:
- Distinguish between different types of conceptual errors
- Provide consistent and efficient tagging
- Generalize across different mathematical problems

### Competition Details

- **Host:** Vanderbilt University & The Learning Agency (in partnership with Kaggle)
- **Data Provider:** Eedi (edtech platform for students ages 9-16)
- **Duration:** July 10, 2025 - October 15, 2025
- **Evaluation Metric:** Mean Average Precision @ 3 (MAP@3)

---

## 🏆 Solution Results

### Final Standings
- **Public Leaderboard:** 0.949 MAP@3
- **Private Leaderboard:** 0.947 MAP@3
- **Rank:** 45th place / Silver Medal 🥈

### Approach Summary
- **6-model LLM ensemble** with intelligent voting
- **Family-based filtering** + multi-factor scoring
- Mix of LoRA, QLoRA, and full fine-tuning strategies
- Diverse model architectures (Qwen, Hunyuan, DeepseekMath)

---

## 🤖 Model Ensemble

| # | Model | Public LB | Training Method | GPU Setup | LoRA Config | Learning Rate | Epochs | Training Time |
|---|-------|-----------|-----------------|-----------|-------------|---------------|--------|---------------|
| 1 | **Hunyuan 7B Instruct** | 0.945 | LoRA | 2×L4 | R:64, α:128 | 2e-4 | 3 | 4h 34m |
| 2 | **Qwen 3 4B** | 0.945 | LoRA | 2×L4 | R:512, WD:0.3 | 2e-5 | 3 | 2h |
| 3 | **Qwen 2 8B** | 0.945 | QLoRA 4-bit | 2×L4 | R:64, α:32 | 2e-4 | 2 | 4h 23m |
| 4 | **Qwen 3 14B** | 0.944 | QLoRA 4-bit | 4×L4 | R:16, α:32 | 2e-4 | 3 | 11h 34m |
| 5 | **Qwen 3 8B** | 0.943 | Full Fine-tune | 4×L4 | N/A | 2e-5 | 3 | 3h |
| 6 | **DeepseekMath 7B** | 0.944 | Full Fine-tune | 4×L4 | N/A | 2e-5 | 3 | 3h 24m |

**Total Training Time:** ~30+ hours  
**Inference Time:** ~6-7 hours for full test set

---

## 📊 Solution Architecture

### 1. Data Preprocessing Pipeline

The preprocessing was consistent across all models:

```python
# Create target labels
train['target'] = train.Category + ":" + train.Misconception
train['label'] = le.fit_transform(train['target'])

# Identify correct answers
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1

# Merge 'is_correct' flag into training data
train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
```

### 2. Input Format

Each training example was formatted as:
```
Question: {QuestionText}
Answer: {MC_Answer}
Is Correct Answer: {Yes/No}
Student Explanation: {StudentExplanation}
```

This structure provides the model with full context: the question, the student's answer choice, whether it's correct, and their reasoning.

---

## 🔧 Training Strategies

### LoRA (Low-Rank Adaptation)
- **Models:** Hunyuan 7B, Qwen 3 4B
- **Advantages:** Memory efficient, faster training, prevents catastrophic forgetting
- **Configuration:** Experimented with ranks from 64 to 512

### QLoRA (Quantized LoRA)
- **Models:** Qwen 2 8B, Qwen 3 14B
- **4-bit NF4 quantization** with bfloat16 compute dtype
- Enabled training of larger models on limited GPU resources

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### Full Fine-tuning
- **Models:** Qwen 3 8B, DeepseekMath 7B
- Complete parameter updates for maximum model adaptation
- Required more GPU memory but achieved strong performance

### Key Hyperparameters
```python
num_train_epochs = 3
learning_rate = 2e-4 to 2e-5  # Varied by model
lr_scheduler_type = "cosine"
warmup_ratio = 0.1
gradient_accumulation_steps = 16
fp16/bf16 = True
```

---

## 🎲 Ensemble Methodology

The ensemble was the critical component that boosted performance from individual 0.943-0.945 scores to 0.949 on public LB.

### Probability Extraction
```python
# Generate full probability distributions
logits = trainer.predict(test_dataset).predictions
probs = softmax(logits, axis=1)

# Save top-K classes with probabilities
top_indices = np.argsort(-probs, axis=1)
for i in range(num_classes):
    prob_dict[f"prob_{i}"] = probs[i, top_indices[i]]
```

### Family-Based Filtering
Key insight: Predictions must respect the category family (True vs False)

```python
# Build family map from correct answers
fam_map = test_df.merge(correct_answers, on=['QuestionId','MC_Answer'])
fam_map['family'] = fam_map['is_correct'].map({1: 'True_', 0: 'False_'})
```

### Weighted Voting with Multi-Factor Scoring

The ensemble combined three scoring components:

1. **Weighted Probability Sum (34%)**: Aggregate model confidence
2. **Agreement Bonus (33%)**: How many models agree on this class
3. **Confidence Bonus (33%)**: Maximum confidence from any model

```python
for class_name in all_classes:
    base_score = class_total_prob[class_name]
    agreement_bonus = class_votes[class_name] / n_models
    confidence_bonus = class_max_prob[class_name]
    
    final_scores[class_name] = (
        base_score * 0.34 +
        agreement_bonus * 0.33 +
        confidence_bonus * 0.33
    )
```

### Family Filtering & Backfilling
```python
# Filter by family
final_scores = {k: v for k, v in final_scores.items() 
                if k.startswith(family_prefix)}

# Backfill if < 3 predictions
fillers = [f"{family}_Neither:NA"]
if family == "True_":
    fillers.append(f"{family}_Correct:NA")
```

---

## 💡 Key Insights & Learnings

### ✅ What Worked

1. **Diversity Matters**
   - Combining different architectures (Qwen, Hunyuan, DeepseekMath) captured complementary patterns
   - Different training strategies (LoRA, QLoRA, full fine-tuning) provided diverse perspectives

2. **Family Constraint is Critical**
   - Enforcing category family (True/False) consistency prevented nonsensical predictions
   - Simple rule-based filtering provided huge gains

3. **Probability Calibration**
   - Saving full probability distributions (not just top-3) enabled sophisticated ensembling
   - Multi-factor scoring (probability + agreement + confidence) outperformed simple averaging

4. **Efficient Training**
   - LoRA and QLoRA enabled training large models (up to 14B parameters) on accessible hardware
   - 4-bit quantization had minimal impact on final performance

5. **Transfer Learning**
   - Using pre-trained models from previous Eedi competition provided a strong starting point
   - Domain-specific pre-training on math misconception data was highly valuable

### ❌ What Didn't Work

**Models Explored but Not Included:**
- OpenAI GPT 20B (training complexity outweighed benefits)
- Generative training approach (text generation inferior to classification)
- Small models: smoLM2/3 (360M-3B parameters lacked capacity)
- 50+ other models (Llama 3.1 8B, Phi 4 12B, etc.)
- Qwen 32B (insufficient time for inference in 4-bit) (I couldn't optimize the code to run on two T4s within the 9-hour time constraint)

**Techniques That Underperformed:**
- Heavy prompt engineering
- Unequal ensemble weights
- Single model approaches

### 🔄 Future Improvements

1. **Hyperparameter Optimization:** More systematic tuning of LoRA ranks, learning rates, and ensemble weights
2. **Data Augmentation:** Synthetic misconception generation to increase training diversity
3. **Learned Ensemble Weights:** Rather than equal weighting
4. **Better Cross-Validation:** Improved validation strategy to detect overfitting

---

## 📁 Repository Structure

```
├── README.md                          # This file
├── writeup.md                         # Detailed writeup
└── training scripts/
    ├── Deepseek Math 7B-.944.py      # DeepseekMath training
    ├── Hunyaun 7B-0.945.py          # Hunyuan training
    ├── Qwen 2-0.945.py              # Qwen 2 training
    ├── Qwen 3 14B-0.944.py          # Qwen 3 14B training
    ├── Qwen 3 4B-0.945.py           # Qwen 3 4B training
    └── Qwen 3 8B-0.944.py           # Qwen 3 8B training
```

---

## 🙏 Acknowledgments

This Silver Medal achievement would not have been possible without the generous contributions of the Kaggle community:

**Competition & Platform:**
- **Kaggle** for providing this incredible platform to learn and compete
- **Competition Hosts** (Vanderbilt University & The Learning Agency) for organizing the MAP competition
- **Eedi** for providing the data
- **Gates Foundation** and **Walton Family Foundation** for supporting this work

**Community Contributors:**
- **[Chris Deotte](https://www.kaggle.com/cdeotte)** - Ettin-Encoder-1B discussion and data preprocessing approach
- **[Kishan Vavdara](https://www.kaggle.com/kishanvavdara)** - Ensemble notebook and multi-model voting strategy
- **[Raja Biswas](https://www.kaggle.com/conjuring92)** - Pre-trained Eedi CoT 14B model

This is my first Silver Medal, and it's a testament to how much you can achieve by standing on the shoulders of giants. Thank you all! 🚀

---

## 📚 Resources

- **Competition:** [MAP - Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)
- **Data Preprocessing:** [Chris Deotte's Ettin-Encoder-1B Discussion](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/590326)
- **Ensemble Technique:** [Kishan Vavdara's Ensemble Notebook](https://www.kaggle.com/code/kishanvavdara/ensemble-gemma-qwen-deepseek?scriptVersionId=260098381&cellId=9)
- **Pre-trained Model:** [Raja Biswas's Eedi CoT 14B](https://www.kaggle.com/models/conjuring92/eedi-cot-14b-dec6-awq/)
- **Libraries:** HuggingFace Transformers, PEFT, BitsAndBytes, scikit-learn

---

## 📊 Evaluation Metric

Submissions were evaluated using **Mean Average Precision @ 3 (MAP@3)**:

$$MAP@3 = \frac{1}{U} \sum_{u=1}^{U} \sum_{k=1}^{\min(n,3)} P(k) \times rel(k)$$

Where:
- $U$ is the number of observations
- $P(k)$ is the precision at cutoff $k$
- $n$ is the number of predictions submitted per observation
- $rel(k)$ is an indicator function equaling 1 if the item at rank $k$ is a relevant (correct) label, zero otherwise

---

## 📝 License

This project is part of a Kaggle competition. Please refer to the competition rules and data usage terms.

---

**Questions or feedback?** Feel free to open an issue or reach out! 🚀
