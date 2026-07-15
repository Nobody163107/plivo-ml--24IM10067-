# Notes

The model predicts whether a pause corresponds to the end of a speaker's turn using only information available before the pause.

Features are extracted from a 1.5-second causal speech window and include energy statistics, pitch statistics, voiced-frame ratio, speech duration, and MFCC mean and standard deviation coefficients.

A StandardScaler followed by Logistic Regression was selected after comparing several approaches.

Gradient Boosting achieved excellent training performance but overfit held-out turns, so it was rejected.

The model performs well when prosodic cues such as falling energy or pitch indicate the end of an utterance.

Errors still occur for ambiguous pauses where speakers hesitate or continue speaking with end-of-sentence prosody.

If given another day, I would add speaking-rate features, final-syllable lengthening, delta MFCCs, and tune hyperparameters using grouped cross-validation.

The implementation is fully causal and never uses information occurring after pause_start.