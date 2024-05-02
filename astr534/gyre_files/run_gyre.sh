#!/bin/bash
# runs GYRE:
# run_gyre.sh PREFIX 0
# will run adiabatic gyre on:
# "$outfile"*.h5 mode files in ./
# using the parameter file gyre_ad.in.temp
# run_gyre.sh PREFIX 1
# will do the same only with non-adiabatic calculation, and using gyre_nonad.in.temp

################################################################
# this will cause the exit status of the entire program to be 1 (i.e., failure) if one of the MESA files fails to
# run because of a floating point error. this error may occur when there are -Infinity or NaN entries in the MESA files.
# otherwise, the exit status will be 0, indicating success.
################################################################
exit=0
sigfpe()
{
   echo "signal FPE received --- skipping file that caused this error."
   exit=1
   exit 0

}

trap sigfpe FPE
################################################################

# JCZ 050418
# the files that will be looked for to run will be $outfile0.* (e.g., $outfile01_0010, $outfile02_0102)
# JCZ 221018
# note that the NN in NN_#### is the run number. for most cases, only 02_#### will be output because the first model is a re-scaling model run, and will move nowhere in the HR diagram far enough to trigger a writing out under LMOUT=.TRUE.
# and the output files will be:
# $outfile.gyre_ad.eigval.h5
# and 
# "$outfile"mode-0001.h5
# etc.

outfile=$1
nonad=$2
# JCZ 221018
# added these options
lsummary=$3
lmode=$4
# JCZ 221018
# for some reason, YREC will set LNONAD to 1 after the rescaling model run (01), and so the actual important run (02) is nad. and that code doesn't work quite right yet. so forcing to be 0. !!! debug
# lsummary and lmode also seem to flip upon the second model... so setting them to 1 for now...

lsummary=1
lmode=1

outdir=$5

if [ "$outdir" -eq "" ]; then
    outdir='./'
fi

echo $outdir
echo 'hi'
echo $lsummary
echo $lmode
if [ "$lsummary" -eq "1" ]; then
    lsummary=''
else
    lsummary='!'
fi

if [ "$lmode" -eq "1" ]; then
    lmode=''
else
    lmode='!'
fi
echo "lmode is $lmode"
echo "lsummary is $lsummary"

################################################################
# clear previous output files
echo "clearing out old mesa files."
rm gyre.err
rm gyre.out
echo "cleared out old mesa files."
################################################################
exe=$GYRE_DIR/bin/gyre

template=./gyre.in.temp

echo $exe

files="${outfile}"*GYRE
echo "$outfile"

# if don't do this, the for loop fails to provide the first file and instead takes files as literally files, with the asterisk.
# files=( $files )

new_file=$PWD/gyre.in
echo "Using $template as the GYRE input file."

################################################################
# run GYRE on each file output by YREC with the LMESA toggle put on.
# this will either be every model (not a good idea), or for models separated in the HR diagram, by
# specifying the LMOUT flag.
################################################################

# JCZ 221018
# it just does the first file. but this is OK, since all the old files are deleted in the clearing out, below. so "${outfile}0*" should only find that newest mesa output file.
for file in ${files}; do
# JCZ 221018
    # have to put this in here for reasons.
if [[ $file =~ 'h5' ]]; then
    continue
fi
if [[ $file =~ 'astero.in' ]]; then
    continue
fi
template_tmp=$template
echo "Running $file in GYRE using the command:"
echo "$exe" "$new_file"
dest=$(dirname ${file})/run_$(basename ${file})

echo "copying the input FGONG file to $dest to indicate that it has been run/ attempted to have been run"
cp ${file} $outdir/${dest}

################################################################
# update the iso template to have the correct initial parameters
################################################################

# adding outdir

slots=( FILEIN FILEOUT LSUMMARY LMODE )
values=( $file $outdir/$file $lsummary $lmode )

nslots=`echo ${#slots[*]} -1 | bc -l`
for i in `seq 0 $nslots`; do
a=${slots[$i]}
b=${values[$i]}
sed "s|$a|$b|" $template_tmp > tmp
cat tmp > $new_file
# JCZ 221018
# need to do this separately for every file one wants to run, and need to start with a fresh template, so template is reset after this loop for the next file using template_tmp. it is necessary here because FILEIN and FILEOUT have to be 
# modified in series
template_tmp=$new_file
done
adflag='ad'
if [ ! "$nonad" -eq "0" ]; then
    echo "doing nonad calc."
    sed "s|! nonadiabatic = .FALSE.|nonadiabatic = .TRUE.|" $template_tmp > tmp
    cat tmp > $new_file
    adflag='nad'
fi

################################################################


################################################################
# then run gyre!
# the time taken to do the gyre run is output to gyre.out, in addition to the frequencies that were found and the input files (i.e., $newfile), among other useful info.
# as far as i can tell, gyre will not send error information --- it will only fail catastrophically 
################################################################
echo $exe
echo $new_file
$exe $new_file #2>> ${outdir}/gyre.err 1>> ${outdir}/gyre.out
#python break_mesa.py $file $adflag --dir $outdir
#echo "appending ${file}.${adflag}.astero.in to ${outdir}/${outfile}.astero"
#cat ${file}.${adflag}.astero.in >> ${outdir}/${outfile}.astero
################################################################
done
# clear previous output files
# echo "clearing out the mesa file that was just run and all ones older than it"
# for file in ${files[0]}; do
#     if ! [[ $file =~ 'h5' ]]; then
# 	# JCZ 080219
# 	# instead of removing the file, move to a directory called gyre_output
	
# 	echo ""

#     fi
#     if [[ $file =~ 'h5' ]]; then
# 	rm $file
# 	mv $file gyre_output
#     fi
# done

exit $exit
