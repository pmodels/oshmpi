# -*- Makefile -*-
#
# Copyright 2011 Sandia Corporation. Under the terms of Contract
# DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
# retains certain rights in this software.
# 
# This file is part of the Portals SHMEM software package. For license
# information, see the LICENSE file in the top level directory of the
# distribution.
#

CC      = mpicc
CFLAGS  = -g -O2 -std=c99 -Wall -I.

LIBRARY = libsma.la

HEADERS = \
	shmem_free_list.h \
	shmem_atomic.h \
	shmem_internal.h \
	shmem_comm.h \
	shmem_collectives.h \
	shmem_synchronization.h \
	shmem_accessibility.h \
	shmem_remote_pointer.h \
	shmem_lock.h \

SOURCES = \
	shmem_free_list.c \
	malloc.c \
	init.c \
	init_c.c \
	query_c.c \
	accessibility_c.c \
	symmetric_heap_c.c \
	remote_pointer_c.c \
	data_c.c \
	atomic_c.c \
	synchronization_c.c \
	collectives_c.c \
	lock_c.c \
	cache_management_c.c \
	init_f.c \
	query_f.c \
	accessibility_f.c \
	symmetric_heap_f.c \
	remote_pointer_f.c \
	data_f.c \
	atomic_f.c \
	synchronization_f.c \
	collectives_f.c \
	lock_f.c \
	cache_management_f.c \


OBJECTS = $(SOURCES:.c=.o)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

transport_mpi3.o: transport_mpi3.c transport_mpi3.h
	$(CC) $(CFLAGS) -c $< -o $@

# transport_mpi3.o

$(LIBRARY): $(OBJECTS) 

clean:
	-rm -f $(OBJECTS)

realclean: clean
	-rm -f $(LIBRARY)

