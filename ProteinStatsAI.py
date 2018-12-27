#protein stats: ftp://ftp.ebi.ac.uk/pub/databases/pombase/pombe/Protein_data/PeptideStats.tsv
#protein fasta: ftp://ftp.ebi.ac.uk/pub/databases/pombase/FASTA/pep.fa.gz


from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np
from time import sleep
import h5py
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

#path to folder 
#path="/ProteinStatsAI/"

#seed for random number reproduction
np.random.seed(2016)

#ngrams range
#minimum range
mi=1
#maximum range
mx=1
#epoch
epoch=15
#batch size
bs=200

#number of cross validation
ncv=1


# read file into pandas from the working directory
data=pd.read_csv("newpep.csv")

X=data.sequence
X.head()
Y=data.ix[:,5:10]
Y.head()

'''
#Y.mean()
Mass           52382.426389
pI                11.470424
Charge             2.632752
NumResidues      464.420019
CAI                0.472803

#Y.max()
Mass           559849.08
pI              21088.02
Charge             74.00
NumResidues      4924.00
CAI               184.00

#Y.min()
Mass           117.0000
pI               3.3614
Charge        -142.5000
NumResidues      0.0000
CAI              0.1650

'''

#normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(Y)
Y_norm = pd.DataFrame(np_scaled)
Y_norm.columns=list(Y)
Y_norm.shape
Y_norm=Y_norm.values

#empty variables
lscore=[]


#custom made k fold cv
#kfold cross validation loop
for i in range(0,ncv):
    #spliting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y_norm,test_size=0.3, random_state=i)
    

    # instantiate the vectorizer    
    vect = CountVectorizer(analyzer=u'char',lowercase=False,ngram_range=(mi,mx))

    # learn training data vocabulary, then use it to create a document-term matrix
    X_train_dtm = vect.fit_transform(X_train)
    X_train_dtm=X_train_dtm.astype('float64')
    X_shape=X_train_dtm.shape

    vect.vocabulary_

    #cnn model
    model = Sequential()
    model.add(Dense(X_train_dtm.shape[1], input_dim=X_train_dtm.shape[1], init='normal', activation='relu'))
    model.add(Dense(20, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1],init='normal', activation='sigmoid'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='mse', optimizer=optimizer)
    
    # transform testing data (using fitted vocabulary) into a document-term matrix
    X_test_dtm = vect.transform(X_test)
    X_test_dtm

    print(model.input_shape)
    print(model.output_shape)
    
    # Fit the model
    history=model.fit(X_train_dtm.toarray(), y_train,validation_data=(X_test_dtm.toarray(),y_test), shuffle=True, nb_epoch=epoch, batch_size=bs)
    
    #accuracy,loss calculation
    los=np.mean(np.array(history.history['val_loss']))
    lscore=np.append(los,lscore)
    

#average
lavg=np.mean(lscore)

#summary
print(vect.vocabulary_)
print(model.summary())
print("\nAverage Validation Loss: " + str(lavg)+"\nLast Validation Loss: " + str(history.history['val_loss'][epoch-1]))








