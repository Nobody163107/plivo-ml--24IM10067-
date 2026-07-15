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

## Experiment 1: Prosodic Feature Engineering

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