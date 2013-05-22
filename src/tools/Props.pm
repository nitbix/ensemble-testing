use Config::Tiny;
use DBI;

use Data::Dumper;

sub open_db {
	my $file = shift;
	my $config = Config::Tiny->read($file);
	my $dbdata = $config->{_};
	die "Cannot open $file" unless (-f $file);	
	return DBI->connect("DBI:mysql:$dbdata->{dbname};host=$dbdata->{dbhost}", $dbdata->{dbuser}, $dbdata->{dbpass});
}

1;
