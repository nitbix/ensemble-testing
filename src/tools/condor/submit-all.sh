#!/bin/bash
for i in magic haberman letterrecognition landsat ionosphere; do condor_submit ${i}.csub; done
