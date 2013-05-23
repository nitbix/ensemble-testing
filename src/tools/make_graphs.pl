#!/usr/bin/perl

use strict;
use warnings;
use DBI;
use Data::Dumper;
use lib '.';
use Props;

my $db = open_db($ARGV[0]);

my $techniques = $db->selectcol_arrayref("select distinct technique from chains where invalidated = 0");

foreach my $technique (@{$techniques}) {
	print "$technique\n";
	my $problems = $db->selectall_arrayref("select problem, count(*) from chains where invalidated = 0 and technique = '$technique' group by problem");
	foreach my $problem (@{$problems}) {
		my $problem_name = $problem->[0];
		print "    $problem->[0] has $problem->[1] chains\n";
		my $stats_by_size = $db->selectall_arrayref(qq{
			select ensemble_size, count(*), avg(misclassified_samples), avg(macro_precision), avg(macro_recall)
			from runs r join chains c on r.chain = c.id
			where invalidated = 0 and problem = '$problem->[0]' 
			and technique = '$technique' and is_test = 1
			group by problem, technique, ensemble_size}
		);
		my $out_file = "output/${technique}-$problem->[0]";
		open OUT, "> ${out_file}.dat";
		foreach my $stats (@{$stats_by_size}) {
			print OUT "$stats->[0] $stats->[1] $stats->[2] $stats->[3] $stats->[4]\n";
		}
		close OUT;
		open GNUPLOT, "| gnuplot";
		print GNUPLOT qq{
			set style line 1 lc rgb '#8b1a0e' lw 2 # --- red
			set style line 2 lc rgb '#5e9c36' lw 2 # --- green
			set style line 3 lc rgb '#993300' lw 2 # --- orange 
			set style line 4 lc rgb '#003366' lw 2 # --- blue
			set style line 11 lc rgb '#808080' lt 1
			set border 3 back ls 11
			set tics nomirror
			set style line 12 lc rgb '#808080' lt 0 lw 1
			set grid back ls 12
#			set title "Misclassification Rates ($technique - $problem_name)"
			set terminal postscript eps enhanced color size 4,2.5
#			set xlabel "Ensemble Size"
#			set ylabel "Error"
			set out "${out_file}-misclass.eps"
			plot "${out_file}.dat" using 1:3 smooth sbezier lw 2 title "Misclassification"
			set terminal png
			set out "${out_file}-misclass.png"
			replot
			set terminal postscript eps enhanced color size 4,2.5			
			set out "${out_file}-precrec.eps"
			plot "${out_file}.dat" using 1:4 smooth sbezier lw 2 title "Precision", "${out_file}.dat" using 1:5 smooth sbezier lw 2 title "Recall"
			set terminal png
			set out "${out_file}-precrec.png"
			replot
		};
		close GNUPLOT;
	}
}

$db->disconnect();