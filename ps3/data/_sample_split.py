import hashlib
import numpy as np
import random

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.
    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    
    #adding a section to be able to handle multiple columns
    

    def assign_split(id_value):
        
        id_value = '_'.join(id_value)
        
        #take some id value and ensure it's a string
        id_string = str(id_value)

        #then convert to byte which is what hashlibs wants
        id_bytes = id_string.encode()

        # create a hash using the the bytes - the hash is some long hex string
        hash_object = hashlib.md5(id_bytes)

        # then we convert the hash (which is a hex string made up of letters and numbers) into a long interger
        hash_int = int(hash_object, 16)

        #then we assign the hash int into a bucket - % 100 taks the last 2 numbers, and this is a good
        #enough approach because the hash distribution is so large and random the distirbution should be
        # approximately uniform - note for small sample doesnt guarantee a perfect split.
        bucket = hash_int % 100

        # then we assign use the bucket to assign what value we want
        threshold = int(training_frac * 100)
        if bucket < threshold:
            return 'train'
        else:
            return 'test'
        
    # now we need to call on the function to be applied to the data frame
    # we do this by applying it to each row, using the apply function
    df['sample'] = df['id_column'].apply(assign_split)

    return df
