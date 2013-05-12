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
		print "    $problem->[0] has $problem->[1] chains\n";
		my $stats_by_size = $db->selectall_arrayref(qq{
			select ensemble_size, count(*), avg(misclassified_samples), avg(macro_precision), avg(macro_recall)
			from runs r join chains c on r.chain = c.id
			where invalidated = 0 and problem = '$problem->[0]' 
			and technique = '$technique' and is_test = 1
			group by problem, technique, ensemble_size}
		);
		open OUT, "> output/${technique}-$problem->[0].dat";
		foreach my $stats (@{$stats_by_size}) {
			print OUT "$stats->[0] $stats->[1] $stats->[2] $stats->[3] $stats->[4]\n";
		}
		close OUT;
	}
}

$db->disconnect();