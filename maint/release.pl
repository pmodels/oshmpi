#!/usr/bin/env perl
#
# (C) 2008 by Argonne National Laboratory.
#     See COPYRIGHT in top-level directory.
#

use strict;
use warnings;

use Cwd qw( cwd getcwd realpath );
use Getopt::Long;
use File::Temp qw( tempdir );

my $arg = 0;
my $branch = "";
my $version = "";
my $append_commit_id;
my $root = cwd();
my $with_autoconf = "";
my $with_automake = "";
my $git_repo = "";
my $with_mpi = "";

my $logfile = "release.log";

sub usage
{
    print "Usage: $0 [OPTIONS]\n\n";
    print "OPTIONS:\n";

    print "\t--git-repo           path to root of the git repository (required)\n";
    print "\t--branch             git branch to be packaged (required)\n";
    print "\t--version            tarball version (required)\n";
    print "\t--with-mpi           mpi installation (optional)\n";
    print "\t--append-commit-id   append git commit description (optional)\n";

    print "\n";

    exit 1;
}

sub check_package
{
    my $pack = shift;

    print "===> Checking for package $pack... ";
    if (`which $pack` eq "") {
        print "not found\n";
        exit;
    }
    print "done\n";
}


sub check_autotools_version
{
    my $tool = shift;
    my $req_ver = shift;
    my $curr_ver;

    $curr_ver = `$tool --version | head -1 | cut -f4 -d' ' | xargs echo -n`;
    if ("$curr_ver" ne "$req_ver") {
    print("\tERROR: $tool version mismatch ($req_ver) required\n\n");
    exit;
    }
}

# will also chdir to the top level of the git repository
sub check_git_repo {
    my $repo_path = shift;

    print "===> chdir to $repo_path\n";
    chdir $repo_path;

    print "===> Checking git repository sanity... ";
    unless (`git rev-parse --is-inside-work-tree 2> /dev/null` eq "true\n") {
        print "ERROR: $repo_path is not a git repository\n";
        exit 1;
    }
    # I'm not strictly sure that this is true, but it's not too burdensome right
    # now to restrict it to complete (non-bare repositories).
    unless (`git rev-parse --is-bare-repository 2> /dev/null` eq "false\n") {
        print "ERROR: $repo_path is a *bare* repository (need working tree)\n";
        exit 1;
    }

    print "done\n";
}


sub run_cmd
{
    my $cmd = shift;

    #print("===> running cmd=|$cmd| from ".getcwd()."\n");
    system("$cmd >> $root/$logfile 2>&1");
    if ($?) {
        die "unable to execute ($cmd), \$?=$?.  Stopped";
    }
}

GetOptions(
    "branch=s" => \$branch,
    "version=s" => \$version,
    "append-commit-id!" => \$append_commit_id,
    "with-autoconf" => \$with_autoconf,
    "with-automake" => \$with_automake,
    "git-repo=s" => \$git_repo,
    "with-mpi=s" => \$with_mpi,
    "help"     => \&usage,
) or die "unable to parse options, stopped";

if (scalar(@ARGV) != 0) {
    usage();
}

if (!$branch || !$version) {
    usage();
}

check_package("git");
check_package("autoconf");
check_package("automake");
check_package("libtool");
print("\n");

## IMPORTANT: Changing the autotools versions can result in ABI
## breakage. So make sure the ABI string in the release tarball is
## updated when you do that.
check_autotools_version("autoconf", "2.69");
check_autotools_version("automake", "1.15");
check_autotools_version("libtool", "2.4.6");
print("\n");

my $tdir = tempdir(CLEANUP => 1);
my $local_git_clone = "${tdir}/oshmpi-clone";

# clone git repo
print("===> Cloning git repo... ");
run_cmd("git clone ${git_repo} -b ${branch} --recursive ${local_git_clone}");
print("done\n");

# chdirs to $local_git_clone if valid
check_git_repo($local_git_clone);
print("\n");

my $current_ver = `git show ${branch}:maint/version.m4 | grep OSHMPI_VERSION_m4 | \
                   sed -e 's/^.*\\[OSHMPI_VERSION_m4\\],\\[\\(.*\\)\\].*/\\1/g'`;
if ("$current_ver" ne "$version\n") {
    print("\tWARNING: maint/version does not match user version\n\n");
}

if ($append_commit_id) {
    my $desc = `git describe --always ${branch}`;
    chomp $desc;
    $version .= "-${desc}";
}

# apply patches to submodules
#print("===> Patching submodules... ");
#run_cmd("./maint/apply_patch.bash");

my $expdir = "${tdir}/oshmpi-${version}";

# Clean up the log file
system("rm -f ${root}/$logfile");

# Check out the appropriate branch
print("===> Exporting code from git... ");
run_cmd("rm -rf ${expdir}");
run_cmd("mkdir -p ${expdir}");
run_cmd("git archive ${branch} --prefix='oshmpi-${version}/' | tar -x -C $tdir");
run_cmd("git submodule foreach --recursive \'git archive HEAD --prefix='' | tar -x -C `echo \${toplevel}/\${path} | sed -e s/clone/${version}/`'");
print("done\n");

print("===> Create release date and version information... ");
chdir($local_git_clone);
my $date = `git log -1 --format=%ci`;
chomp $date;

chdir($expdir);
system(qq(perl -p -i -e 's/\\[OSHMPI_RELEASE_DATE_m4\\],\\[unreleased development copy\\]/[OSHMPI_VERSION_m4],[$date]/g' ./maint/version.m4));
print("done\n");

# Remove content that is not being released
print("===> Removing content that is not being released... ");
chdir($expdir);
print("done\n");

# Create configure
print("===> Creating configure in the main codebase... ");
chdir($expdir);
{
    my $cmd = "./autogen.sh";
    run_cmd($cmd);
}
print("done\n");

# Execute configure and make dist
print("===> Running configure... ");
run_cmd("./configure --prefix=$expdir/install CC=$with_mpi/bin/mpicc");
print("done\n");

print("===> Making the final tarball... ");
run_cmd("make dist");
run_cmd("cp -a oshmpi-${version}.tar.gz ${root}/");
print("done\n");

# make sure we are outside of the tempdir so that the CLEANUP logic can run
chdir("${tdir}/..");
