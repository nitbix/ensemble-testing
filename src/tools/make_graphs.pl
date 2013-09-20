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
		print "    $problem_name has $problem->[1] chains\n";
		my $aggregations = $db->selectcol_arrayref("select distinct aggregation from chains where technique = '$technique' and problem = '$problem_name'");
		foreach my $aggregation (@{$aggregations}) {
			print "        ($aggregation)\n";
			my $stats_by_size = $db->selectall_arrayref(qq{
				select ensemble_size, count(*),
				avg(misclassified_samples), avg(micro_precision), avg(micro_recall),
				min(misclassified_samples), min(micro_precision), min(micro_recall),
				max(misclassified_samples), max(micro_precision), max(micro_recall)
				from runs r join chains c on r.chain = c.id
				where invalidated = 0
				and problem = '$problem_name' 
				and technique = '$technique'
				and is_test = 1
				and aggregation = '$aggregation'
				group by problem, technique, ensemble_size}
			);
			my $out_file = "output/${technique}-$problem_name-$aggregation";
			open OUT, "> ${out_file}.dat";
			foreach my $stats (@{$stats_by_size}) {
				print OUT (join " ", @$stats) . "\n";

			}
			close OUT;
			open GNUPLOT, "| gnuplot";
			print GNUPLOT qq{
				set style line 1 lc rgb '#8b1a0e' lw 2 # --- red
				set style line 2 lc rgb '#5e9c36' lw 2 # --- green
				set style line 3 lc rgb '#993300' lw 2 # --- orange 
				set style line 4 lc rgb '#003366' lw 2 # --- blue
				set style line 11 lc rgb '#808080' lt 1
				set style errorbars 1 lc rgb '#8b1a0e' lw 2 # --- red
				set style errorbars 2 lc rgb '#5e9c36' lw 2 # --- green
				set style errorbars 3 lc rgb '#993300' lw 2 # --- orange 
				set style errorbars 4 lc rgb '#003366' lw 2 # --- blue
				set style errorbars 11 lc rgb '#808080' lt 1
				set border 3 back ls 11
				set tics nomirror
				set style line 12 lc rgb '#808080' lt 0 lw 1
				set grid back ls 12
#				set title "Misclassification Rates ($technique - $problem_name)"
				set terminal postscript eps enhanced color size 4,2.5
#				set xlabel "Ensemble Size"
#				set ylabel "Error"
				set out "${out_file}-misclass.eps"
				plot "${out_file}.dat" using 1:3 smooth sbezier lw 2 title "Misclassification", "${out_file}.dat" using 1:3:6:9 with errorbars title ""
				set terminal png
				set out "${out_file}-misclass.png"
				replot
				set terminal postscript eps enhanced color size 4,2.5			
				set out "${out_file}-precrec.eps"
				plot "${out_file}.dat" using 1:4 smooth sbezier lw 2 title "Precision", "${out_file}.dat" using 1:5 smooth sbezier lw 2 title "Recall", "${out_file}.dat" using 1:4:7:10 with errorbars title "", "${out_file}.dat" using 1:5:8:11 with errorbars title ""
				set terminal png
				set out "${out_file}-precrec.png"
				replot
				set out "${out_file}-prec.eps"
				plot "${out_file}.dat" using 1:4 smooth sbezier lw 2 title "Precision", "${out_file}.dat" using 1:4:7:10 with errorbars title ""
				set terminal png
				set out "${out_file}-prec.png"
				replot
				set out "${out_file}-rec.eps"
				plot "${out_file}.dat" using 1:5 smooth sbezier lw 2 title "Recall", "${out_file}.dat" using 1:5:8:11 with errorbars title ""
				set terminal png
				set out "${out_file}-rec.png"
				replot
			};
			close GNUPLOT;
		}
	}
}

foreach my $technique_a (@{$techniques}) {
foreach my $technique_b (@{$techniques}) {
	foreach my $aggregation_a (@{$aggregations}) {
	foreach my $aggregation_b (@{$aggregations}) {
	}
	}
}
}
$db->disconnect();
