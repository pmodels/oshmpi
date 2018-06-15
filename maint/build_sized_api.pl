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

open(SIZEFILE, "$sizefile");

# Move to file beginning
seek SIZEFILE, 0, 0;

my $line = "";
my @sizedefs;
my $x;
my $skip = 1;
my $is_header_footer = 0;

if ( !$append ) {
    open(CFILE, ">$outfile") || die "Could not open $outfile\n";
} else {
    # Open file in append mode
    open(CFILE, ">>$outfile") || die "Could not open $outfile\n";
}

# Print template header once
open(TPLFILE, "$tplfile");
seek TPLFILE, 0, 0;
while(<TPLFILE>)
{
    if (/TPL_HEADER_START/) { 
        $is_header_footer=1;
        next; # print from next line
    }
    if (/TPL_HEADER_END/) { last; }

    if ( $is_header_footer == 1 ) {
        print CFILE $_;
    }
}

while (<SIZEFILE>) {
    # Skip comment lines
    if (/#/) { next; }

    # Read the size definition of each line [TYPE, SIZENAME, MPITYPE]
    @sizedefs = split(/,/, $_);

    # Cleanup white space
    for ($x = 0; $x <= $#sizedefs; $x++) {
        $sizedefs[$x] =~ s/^\s*//g;
        $sizedefs[$x] =~ s/\s*$//g;
    }

    # Move to TPLFILE beginning
    open(TPLFILE, "$tplfile");
    seek TPLFILE, 0, 0;
    $skip=0;

    while(<TPLFILE>)
    {
        # Skip header and footer
        if (/TPL_HEADER_START|TPL_FOOTER_START/) { 
            $skip=1;
        }
        if (/TPL_HEADER_END|TPL_FOOTER_END/) {
            $skip=0;
            next;
        }
        if ( $skip == 1 ) { next; }

        $_ =~ s/SIZENAME/$sizedefs[1]/g;
        $_ =~ s/MPI_TYPE/$sizedefs[2]/g;
        $_ =~ s/SIZE/$sizedefs[0]/g;
        print CFILE $_;
    }
}

# Print template footer once
open(TPLFILE, "$tplfile");
seek TPLFILE, 0, 0;
$is_header_footer=0;

while(<TPLFILE>)
{
    if (/TPL_FOOTER_START/) { 
        $is_header_footer=1;
        next; # print from next line
    }
    if (/TPL_FOOTER_END/) { last; }

    if ( $is_header_footer == 1 ) {
        print CFILE $_;
    }
}
close CFILE;
