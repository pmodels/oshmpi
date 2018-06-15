#! /usr/bin/env perl
#
# (C) 2018 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

use warnings;
use strict;
use Getopt::Long;

my $typefile = "";
my $tplfile = "";
my $outfile = "";
my $append = 0;

sub usage
{
    print "Usage: $0 --typefile [type definition file] 
           --tplfile [source template file] --outfile [output file] --append\n";
    exit 1;
}

GetOptions(
    "typefile=s" => \$typefile,
    "tplfile=s" => \$tplfile,
    "outfile=s" => \$outfile,
    "append" => \$append
    ) or die "unable to parse options, stopped";

if (!$typefile || !$tplfile || !$outfile) {
    usage();
}

open(TYPEFILE, "$typefile");

# Move to file beginning
seek TYPEFILE, 0, 0;

my $line = "";
my @typedefs;
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

while (<TYPEFILE>) {
    # Skip comment lines
    if (/#/) { next; }

    # Read the type definition of each line [TYPE, TYPENAME, MPITYPE]
    @typedefs = split(/,/, $_);

    # Cleanup white space
    for ($x = 0; $x <= $#typedefs; $x++) {
        $typedefs[$x] =~ s/^\s*//g;
        $typedefs[$x] =~ s/\s*$//g;
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

        $_ =~ s/TYPENAME/$typedefs[1]/g;
        $_ =~ s/MPI_TYPE/$typedefs[2]/g;
        $_ =~ s/TYPE/$typedefs[0]/g;
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
