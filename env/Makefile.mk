#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#

bin_SCRIPTS += env/oshcc 
CLEANFILES += env/oshcc

if OSHMPI_HAVE_CXX
bin_SCRIPTS += env/oshc++
CLEANFILES += env/oshc++
endif

EXTRA_DIST += env/oshcc.in env/oshc++.in 

do_subst = sed  -e 's|[@]OSHMPI_CC[@]|$(CC)|g' \
		-e 's|[@]OSHMPI_CXX[@]|$(CXX)|g' \
		-e 's|[@]OSHMPI_FC[@]|$(FC)|g' \
		-e 's|[@]OSHMPI_INCDIR[@]|$(includedir)|g' \
		-e 's|[@]OSHMPI_LIBDIR[@]|$(libdir)|g' \
		-e 's|[@]WRAPPER_LDFLAGS[@]|$(WRAPPER_LDFLAGS)|g' \
		-e 's|[@]WRAPPER_LIBS[@]|$(WRAPPER_LIBS)|g'

AM_V_SED = $(am__v_SED_@AM_V@)
am__v_SED_ = $(am__v_SED_@AM_DEFAULT_V@)
am__v_SED_0 = @echo "  SED     " $@;

env/oshcc: $(top_srcdir)/env/oshcc.in Makefile
	@mkdir -p env
	$(AM_V_SED)$(do_subst) -e 's|[@]LANG[@]|C|g' < $(top_srcdir)/env/oshcc.in > $@
	@chmod +x $@
	
env/oshc++: $(top_srcdir)/env/oshc++.in Makefile
	@mkdir -p env
	$(AM_V_SED)$(do_subst) -e 's|[@]LANG[@]|C|g' < $(top_srcdir)/env/oshc++.in > $@
	@chmod +x $@