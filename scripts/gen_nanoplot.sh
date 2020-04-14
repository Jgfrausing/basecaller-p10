usage=""$0" [-h/--help] ref.fasta pred.fasta [-o/--open] -- generate a nanoplot folder using two fasta or fastq files.

where:
  -h          shows this help text
  --help      --|--
   
  -o          open nanoplot website once finished
  --open      --|--

dependencies:
  minimap2
  samtools
  NanoPlot"

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  echo "$usage"
  exit 0
fi

mappings_file="$2_mappings.bam"
nanoplot_folder="$2_nanoplot"

minimap2 -t 20 -ax map-ont $1 $2 | samtools view --threads 5 -Sb -F 0x104 - | samtools sort --threads 19 - > "$mappings_file"

# Note the maxlength parameter
NanoPlot --maxlength 400 -t 5 --bam "$mappings_file" --outdir "$nanoplot_folder" 

# Clean up
rm "$mappings_file"
rm "$mappings_file.bai"

echo "Saved results in folder: $nanoplot_folder"

if [ "$3" == "-o" ] || [ "$3" == "--open" ]; then
  open "$nanoplot_folder/NanoPlot-report.html"
fi
