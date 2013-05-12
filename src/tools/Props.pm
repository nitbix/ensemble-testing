use Config::Tiny;
use DBI;

sub open_db {
	my $file = shift;
	my $config = Config::Tiny->read($file);
	my $dbdata = $config->{_};
	return DBI->connect("DBI:mysql:$dbdata->{dbname}:$dbdata{dbhost}", $dbdata->{dbuser}, $dbdata{dbpass});
}

1;
