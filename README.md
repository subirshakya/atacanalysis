# atacanalysis
ATAC seq mapping to reference genome. Gathering 45 ATAC datasets (120 total including replicates) and mapping them to galgal7b

get_sra > download SRA data
fasterq-dump > convert SRA to fastq
NGmerge > remove adapter contamination
map2galgal > uses bwa to match fastq files to galgal7b
Genrich > call atac peaks
