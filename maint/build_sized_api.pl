#! /usr/bin/env perl
#
# (C) 2018 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

use warnings;
use strict;
use Getopt::Long;

my $sizefile = "";
my $tplfile = "";
my $outfile = "";
my $append = 0;

sub usage
{
    print "Usage: $0 --sizefile [size definition file]
           --tplfile [source template file] --outfile [output file] --append\n";
    exit 1;
}

GetOptions(
    "sizefile=s" => \$sizefile,
    "tplfile=s" => \$tplfile,
    "outfile=s" => \$outfile,
    "append" => \$append
    ) or die "unable to parse options, stopped";

if (!$sizefile || !$tplfile || !$outfile) {
    usage();
}

my @allsizedefs;
my @sizedefs;
my $x;
my $nsizes;
my $start = 0;
my $start_pos = 0;
my $newline = "";

if ( !$append ) {
    open(CFILE, ">$outfile") || die "Could not open $outfile\n";
} else {
    # Open file in append mode
    open(CFILE, ">>$outfile") || die "Could not open $outfile\n";
}

# Read size file
open(SIZEFILE, "$sizefile");
seek SIZEFILE, 0, 0;
$nsizes=0;
while (<SIZEFILE>) {
    # Skip comment lines
    if (/#/) { next; }

    # Read the size definition of each line [SIZE, SIZENAME, MPITYPE]
    @sizedefs = split(/,/, $_);

    # Cleanup white space
    for ($x = 0; $x <= $#sizedefs; $x++) {
        $sizedefs[$x] =~ s/^\s*//g;
        $sizedefs[$x] =~ s/\s*$//g;
    }
    $allsizedefs[$nsizes] = [ @sizedefs ];
    $nsizes++;
}
close SIZEFILE;

# Read template file
open(TPLFILE, "$tplfile");
seek TPLFILE, 0, 0;
while(<TPLFILE>)
{
    # Check if a replace block starts, record the start position
    if (/TPL_BLOCK_START/) {
        $start=1;
        $start_pos=tell TPLFILE;
        next;
    }

    # Print plain text
    if ($start == 0) { print CFILE $_; next; }

    if ($start == 1) {
        # Generate the block with each size
        for ($x = 0; $x <= $#allsizedefs; $x++) {

            # Move back to block start
            seek TPLFILE, $start_pos, 0;
            while (<TPLFILE>) {
                if (/TPL_BLOCK_END/) { last; }
                $newline = $_;
                $newline =~ s/SIZENAME/$allsizedefs[$x][1]/g;
                $newline =~ s/MPI_TYPE/$allsizedefs[$x][2]/g;
                $newline =~ s/SIZE/$allsizedefs[$x][0]/g;
                print CFILE $newline;
            }
        }
        # End of a block
        $start = 0;
    }
}
close TPLFILE;
close CFILE;
