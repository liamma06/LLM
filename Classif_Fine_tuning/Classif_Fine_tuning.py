def random_split(df, train_frac, validation_frac):

    #shuffle dataset 
    df = df.sample(
        frac=1,#all rows 
        random_state = 123# seed
    ).reset_index(drop=True) #reset index after shuffling

    train_end= int(train_frac * len(df)) #index where training set ends(int to remove decimal)
    validation_end = train_end + int(validation_frac * len(df)) #index where validation set ends

    train_df = df[:train_end] 
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df