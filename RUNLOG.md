## Run 0: Baseline

Model:
- Constant predictor
- p_eot = 1 for all pauses

Results:

English:
- AUC: 0.514
- Delay: 1600 ms

Hindi:
- AUC: 0.501
- Delay: 850 ms

Observation:
The baseline achieves low delay by aggressively predicting EOT,
but has no understanding of speech context.

## Exact log : 

turns=100  pauses=248  AUC=0.514
BEST @ <= 5% interrupted turns:
  mean response delay : 1600 ms   <-- your score, lower is better
  interrupted turns   : 0.0%
  operating point     : threshold=1.0, delay=1600 ms

<br>

turns=100  pauses=248  AUC=0.501
BEST @ <= 5% interrupted turns:
  mean response delay : 850 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.05, delay=850 ms

## Results from Check Distribution: 
   turn_id         audio_file  pause_index  pause_start  pause_end label
0  en__000  audio/en__000.wav            0          6.3      6.600  hold
1  en__000  audio/en__000.wav            1          8.6      9.500  hold
2  en__000  audio/en__000.wav            2         10.9     11.700  hold
3  en__000  audio/en__000.wav            3         13.7     15.071   eot
4  en__001  audio/en__001.wav            0          4.8      5.000  hold
label
hold    148
eot     100
Name: count, dtype: int64
label
hold    148
eot     100
Name: count, dtype: int64

---

## Experiment 1: Prosodic Feature Engineering (English)

### Changes

Replaced the starter feature extractor with a richer handcrafted feature set.

Added features:

- Mean frame energy
- Energy standard deviation
- Peak energy
- Final frame energy
- Energy slope
- Mean pitch (F0)
- Pitch standard deviation
- Final pitch
- Pitch slope
- Voiced frame ratio
- Speech context length

The classifier remained unchanged (Logistic Regression with balanced class weights).

### Results (English)

- Held-out turn accuracy: 0.600
- AUC: 0.599
- Mean response delay: 1190 ms
- Interrupted turns: 5.0%

### Observations

Feature engineering substantially improved the model.

Compared to the baseline:

- AUC improved from 0.514 to 0.599.
- Mean response delay reduced from 1600 ms to 1190 ms while satisfying the interruption constraint.

This indicates that acoustic cues before the pause provide meaningful information for end-of-turn prediction.


## Experiment 1 (Hindi): Enhanced Prosodic Features

### Motivation

Evaluate whether the improved handcrafted prosodic features (energy and
pitch statistics) generalize to the Hindi dataset.

### Changes

Added:

- Mean energy
- Energy standard deviation
- Peak energy
- Final frame energy
- Energy slope
- Mean pitch
- Pitch standard deviation
- Final pitch
- Pitch slope
- Voiced frame ratio
- Speech context duration

Feature vector size increased from 3 to 11.

### Results (Hindi)

Held-out turn accuracy: 0.552

AUC: 0.729

Mean response delay: 850 ms

Interrupted turns: 5.0%

### Conclusion

The enhanced prosodic features generalized well to the Hindi dataset,
providing a significant improvement over the baseline. Pitch and energy
statistics appear to be strong indicators of End-of-Turn detection for
Hindi speech.


## Experiment 2: Added Spectral Features (MFCC) -> (English)

### Motivation

Prosodic features capture loudness and pitch but ignore the spectral
characteristics of speech. MFCCs summarize the short-term spectral envelope
and are widely used in speech recognition tasks.

### Changes

Added:

- 13 MFCC means
- 13 MFCC standard deviations

Feature vector size increased from 11 to 37.

### Results (English)

Held-out accuracy: 0.554

AUC: 0.726

Mean response delay: 1157 ms

Interrupted turns: 4.0%

### Conclusion

MFCC features significantly improved ranking performance (AUC) while
slightly reducing response delay. This suggests spectral cues provide
additional information beyond prosodic features.

## Experiment 2 (Hindi): Added Spectral Features (MFCC)

### Motivation

Prosodic features capture loudness and pitch but ignore the spectral
characteristics of speech. MFCCs summarize the short-term spectral envelope
and are widely used in speech recognition tasks.

### Changes

Added:

- 13 MFCC mean coefficients
- 13 MFCC standard deviation coefficients

Feature vector size increased from 11 to 37.

### Results (Hindi)

Held-out turn accuracy: 0.603

AUC: 0.809

Mean response delay: 753 ms

Interrupted turns: 5.0%

### Notes

Training produced a Logistic Regression convergence warning, indicating
that the optimizer reached the maximum iteration limit. Despite this,
performance improved substantially over the prosodic-only model.

### Conclusion

MFCC features significantly improved both ranking performance (AUC) and
response latency on the Hindi dataset, demonstrating that spectral
information complements prosodic cues for End-of-Turn detection.



## Experiment 3: Classifier Comparison (HistGradientBoosting)

### Motivation

After improving the feature representation with prosodic features and MFCCs,
we evaluated whether a more expressive nonlinear classifier could improve
End-of-Turn detection.

### Change

Replaced Logistic Regression with:

- HistGradientBoostingClassifier
- learning_rate = 0.05
- max_depth = 3
- max_iter = 200

The feature extraction pipeline remained unchanged (37 features).

### Results (English)

Held-out turn accuracy: 0.492

Training-set scorer results:

- AUC: 1.000
- Mean response delay: 100 ms
- Interrupted turns: 2.0%

### Analysis

Although the scorer reported near-perfect performance, the held-out accuracy
decreased compared to the Logistic Regression model.

This indicates that the Gradient Boosting model overfit the training data.
The assignment scorer evaluates predictions generated after refitting on the
entire dataset, which can produce overly optimistic results for high-capacity
models.

### Decision

Rejected.

The final solution retains Logistic Regression because it provides better
generalization while still achieving strong performance after feature
engineering.

---


## Experiment 4: Feature Standardization

### Motivation

The handcrafted feature vector combines heterogeneous features such as energy,
pitch, speech duration and MFCC coefficients, each with different numerical
ranges. Logistic Regression typically performs better when input features are
standardized.

### Changes

Wrapped the classifier in a scikit-learn pipeline:

- StandardScaler
- LogisticRegression (class_weight="balanced", max_iter=3000)

The feature extraction pipeline remained unchanged.

### Results

| Dataset | Held-out Accuracy | AUC | Mean Delay |
|---------|------------------:|----:|-----------:|
| English | 0.585 | 0.745 | 1045 ms |
| Hindi | 0.552 | 0.814 | 850 ms |

### Conclusion

Feature standardization improved optimization and increased AUC on both
datasets without increasing model complexity. The StandardScaler +
LogisticRegression pipeline was selected as the final model.



## Final Selected Model

### Feature Extraction

- 1.5-second causal speech window before each pause
- Energy statistics
- Energy slope
- Pitch statistics
- Pitch slope
- Voiced-frame ratio
- Speech context duration
- 13 MFCC mean coefficients
- 13 MFCC standard deviations

Total feature dimension: 37

### Classifier

Pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=3000
    )
)

### Final  Performance

| Dataset | Held-out Accuracy |       AUC |  Mean Delay |
| ------- | ----------------: | --------: | ----------: |
| English |         **0.585** | **0.745** | **1045 ms** |
| Hindi   |         **0.552** | **0.814** |  **850 ms** |

## Final Inference Pipeline

Implemented a standalone `predict.py` that:

- Loads the trained model from disk using Joblib.
- Extracts the same handcrafted feature vector.
- Produces causal pause-level predictions.
- Writes predictions in the required CSV format.

The inference pipeline reproduces the validation performance obtained during training, ensuring consistency between training and deployment.