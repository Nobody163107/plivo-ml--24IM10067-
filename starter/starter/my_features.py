# This is feature extractor file which extracts the features using the utilities from features.py. 

def extract_features(audio, sr, pause_start, pause_index): 
    '''
    Extracts the essential features like energy, pitch, duration. 
    we initially add the energy, pitch, timing, MFCC, 
    
    '''