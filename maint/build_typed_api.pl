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

my @alltypedefs;
my @typedefs;
my $x;
my $ntypes;
my $start = 0;
my $start_pos = 0;
my $newline = "";
my $start_c11 = 0;

if ( !$append ) {
    open(CFILE, ">$outfile") || die "Could not open $outfile\n";
} else {
    # Open file in append mode
    open(CFILE, ">>$outfile") || die "Could not open $outfile\n";
}

# Read type file
open(TYPEFILE, "$typefile");
seek TYPEFILE, 0, 0;
$ntypes=0;
while (<TYPEFILE>) {
    # Skip comment lines
    if (/#/) { next; }

    # Read the type definition of each line [TYPE, TYPENAME, MPITYPE, C11_INCLDUE]
    @typedefs = split(/,/, $_);

    # Cleanup white space
    for ($x = 0; $x <= $#typedefs; $x++) {
        $typedefs[$x] =~ s/^\s*//g;
        $typedefs[$x] =~ s/\s*$//g;
    }
    $alltypedefs[$ntypes] = [ @typedefs ];
    $ntypes++;
}
close TYPEFILE;

# Read template file
open(TPLFILE, "$tplfile");
seek TPLFILE, 0, 0;
while(<TPLFILE>)
{
    # Check if a replace block starts, record the start position
    if (/TPL_BLOCK_START|TPL_C11_BLOCK_START/) {
        $start=1;
        $start_pos=tell TPLFILE;

        # Mark if it is a common block or C11 block
        if (/TPL_C11_BLOCK_START/) { $start_c11=1;}
        else { $start_c11=0;}
        next;
    }

    # Print plain text
    if ($start == 0) { print CFILE $_; next; }

    if ($start == 1) {
        # Generate the block with each type
        for ($x = 0; $x <= $#alltypedefs; $x++) {
            # Skip a type if it is excluded for C11 block
            if ($start_c11 == 1 && $alltypedefs[$x][3] eq "0") { next; }

            # Move back to block start
            seek TPLFILE, $start_pos, 0;
            while (<TPLFILE>) {
                if (/TPL_BLOCK_END|TPL_C11_BLOCK_END/) { last; }
                $newline = $_;
                $newline =~ s/TYPENAME/$alltypedefs[$x][1]/g;
                $newline =~ s/MPI_TYPE/$alltypedefs[$x][2]/g;
                $newline =~ s/TYPE/$alltypedefs[$x][0]/g;
                print CFILE $newline;
            }
        }
        # End of a block
        $start = 0;
    }
}
close TPLFILE;
close CFILE;