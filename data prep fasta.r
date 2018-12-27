#bio string installation https://bioconductor.org/packages/release/bioc/html/Biostrings.html
#protein stats: ftp://ftp.ebi.ac.uk/pub/databases/pombase/pombe/Protein_data/PeptideStats.tsv
#protein fasta: ftp://ftp.ebi.ac.uk/pub/databases/pombase/FASTA/pep.fa.gz

library("Biostrings")
set.seed(2016)

#path to the folder containing dataset and stats value


#loading protein staistics
p=read.table(file =paste(path,'PeptideStats.tsv', sep = ""), sep = '\t', header = TRUE)

#loading protein fasta file
s = readAAStringSet(paste(path,'pep.fa', sep = ""))

#cleaning sequence ids 
stable_id = names(s)
stable_id= sub('\\|.*', '', stable_id)

#getting sequence
sequence = paste(s)

#converting to dataframe
df <- data.frame(names(s),stable_id, sequence,width(s))
dim(df)
dim(p)

#mergeing two data frame
result=merge(df,p, by=c("stable_id"))

dim(unique(result))

#exporting to CSV file
write.csv(result,file=(paste(path,'newpep.csv', sep = "")))
