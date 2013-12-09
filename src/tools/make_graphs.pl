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
			next unless scalar(@$stats_by_size);
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
				set terminal postscript eps enhanced color size 4,2.5			
				set out "${out_file}-prec.eps"
				plot "${out_file}.dat" using 1:4 smooth sbezier lw 2 title "Precision", "${out_file}.dat" using 1:4:7:10 with errorbars title ""
				set terminal png
				set out "${out_file}-prec.png"
				replot
				set terminal postscript eps enhanced color size 4,2.5			
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

sub make_comparison {
	my $file_1 = shift;
	my $file_2 = shift;
	my $out_file = shift;
	my $title_1 = shift;
	my $title_2 = shift;
	print "generating $file_1 vs $file_1\n";
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
		set terminal postscript eps enhanced color size 4,2.5
		set out "${out_file}-misclass-comparison.eps"
		plot "${file_1}.dat" using 1:3 smooth sbezier lw 2 title "$title_1", "${file_2}.dat" using 1:3 smooth sbezier lw 2 title "${title_2}"
		set terminal png
		set out "${out_file}-misclass-comparison.png"
		replot
	};
	close GNUPLOT;
}

foreach my $technique_a (@{$techniques}) {
foreach my $technique_b (@{$techniques}) {
	my $problems = $db->selectall_arrayref("select problem, count(*) from chains where invalidated = 0 and (technique = '$technique_a' or technique = '$technique_b') group by problem");
	foreach my $problem (@{$problems}) {
		my $problem_name = $problem->[0];
		my $aggregations = $db->selectcol_arrayref("select distinct aggregation from chains where (technique = '$technique_a' or technique = '$technique_b') and problem = '$problem_name'");
		foreach my $aggregation_a (@{$aggregations}) {
		foreach my $aggregation_b (@{$aggregations}) {
			my $file_aa = "output/${technique_a}-$problem_name-${aggregation_a}";
			my $file_ab = "output/${technique_a}-$problem_name-${aggregation_b}";
			my $file_ba = "output/${technique_b}-$problem_name-${aggregation_a}";
			my $file_bb = "output/${technique_b}-$problem_name-${aggregation_b}";
			my $size_aa = -f "$file_aa.dat" ? `wc -l $file_aa.dat | awk '{print \$1}'` : 0;
			my $size_ab = -f "$file_ab.dat" ? `wc -l $file_ab.dat | awk '{print \$1}'` : 0;
			my $size_ba = -f "$file_ba.dat" ? `wc -l $file_ba.dat | awk '{print \$1}'` : 0;
			my $size_bb = -f "$file_bb.dat" ? `wc -l $file_bb.dat | awk '{print \$1}'` : 0;
			make_comparison($file_aa,$file_ab, "output/${problem_name}-${technique_a}-${aggregation_a}-${aggregation_b}", "${aggregation_a}", "${aggregation_b}") if $size_aa and $size_ab and $aggregation_a ne $aggregation_b;
			make_comparison($file_ba,$file_bb, "output/${problem_name}-${technique_b}-${aggregation_a}-${aggregation_b}", "${aggregation_a}", "${aggregation_b}") if $size_ba and $size_bb and $aggregation_a ne $aggregation_b;
			make_comparison($file_aa,$file_ba, "output/${problem_name}-${aggregation_a}-${technique_a}-${technique_b}", "${technique_a}", "${technique_b}") if $size_aa and $size_ba and $technique_a ne $technique_b;
			make_comparison($file_ab,$file_bb, "output/${problem_name}-${aggregation_b}-${technique_a}-${technique_b}", "${technique_a}", "${technique_b}") if $size_ab and $size_bb and $technique_a ne $technique_b;
		}
		}
	}
}
}
$db->disconnect();
