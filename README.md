# atacanalysis
ATAC seq mapping to reference genome. Gathering 45 ATAC datasets (120 total including replicates) and mapping them to galgal7b \n
\n
get_sra > download SRA data \n
fasterq-dump > convert SRA to fastq \n
NGmerge > remove adapter contamination \n
map2galgal > uses bwa to match fastq files to galgal7b \n
Genrich > call atac peaks
