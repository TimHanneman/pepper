//
// Created by Kishwar Shafin on 10/24/18.
//

#ifndef PEPPER_HP_CANDIDATE_FINDER_H
#define PEPPER_HP_CANDIDATE_FINDER_H

#include <cmath>
#include "../dataio/bam_handler.h"
using namespace std;

namespace CandidateFinder_options {
    static constexpr int min_mapping_quality = 1;
    static constexpr int min_base_quality = 1;
    static constexpr int freq_threshold = 2;
    static constexpr int min_count_threshold = 2;
};

namespace AlleleType {
    static constexpr int SNP_ALLELE = 1;
    static constexpr int INSERT_ALLELE = 2;
    static constexpr int DELETE_ALLELE = 3;
};

namespace Genotype {
    static constexpr int HOM = 0;
    static constexpr int HET = 1;
    static constexpr int HOM_ALT = 2;
};

struct CandidateAllele{
    string ref;
    string alt;
    int alt_type;

    CandidateAllele(string ref, string alt, int alt_type) {
        this->ref = ref;
        this->alt = alt;
        this->alt_type = alt_type;
    }

    CandidateAllele() {}
};

struct Candidate{
    long long pos;
    long long pos_end;
    CandidateAllele allele;
    int genotype;
    int depth;
    int read_support;
    int read_support_h0;
    int read_support_h1;
    int read_support_h2;

    Candidate() {
        this->genotype = 0;
    }

    void set_genotype(int genotype) {
        this->genotype = genotype;
    }

    void set_depth_values(int depth, int read_support, int read_support_h0, int read_support_h1, int read_support_h2) {
        this->depth = depth;
        this->read_support = read_support;
        this->read_support_h0 = read_support_h0;
        this->read_support_h1 = read_support_h1;
        this->read_support_h2 = read_support_h2;
    }

    Candidate(long long pos_start, long long pos_end, string ref, string alt, int alt_type) {
        this->pos = pos_start;
        this->pos_end = pos_end;
        this->allele.alt = alt;
        this->allele.ref = ref;
        this->allele.alt_type = alt_type;
        this->genotype = 0;
    }
    bool operator< (const Candidate& that ) const {
        if(this->pos != that.pos) return this->pos < that.pos;
        if(this->pos_end != that.pos_end) return this->pos_end < that.pos_end;
        if(this->allele.alt != that.allele.alt) return this->allele.alt < that.allele.alt;
        if(this->allele.ref != that.allele.ref) return this->allele.ref < that.allele.ref;
        if(this->allele.alt_type != that.allele.alt_type) return this->allele.alt_type < that.allele.alt_type;
        return this->pos < that.pos;
    }

    bool operator==(const Candidate& that ) const {
        if(this->pos == that.pos &&
           this->pos_end == that.pos_end &&
           this->allele.ref == that.allele.ref &&
           this->allele.alt == that.allele.alt &&
           this->allele.alt_type == that.allele.alt_type)
            return true;
        return false;
    }

    void print() {
        cout<<this->pos<<" "<<this->pos_end<<" "<<this->allele.ref<<" "<<this->allele.alt<<" "<<this->allele.alt_type<<" "<<this->genotype<<endl;
    }

};

struct PositionalCandidateRecord{
    string chromosome_name;
    long long pos_start;
    long long pos_end;
    int depth;
    vector<Candidate> candidates;

    PositionalCandidateRecord(string chromosome_name, long long pos_start, long long pos_end, int depth) {
        this->chromosome_name = chromosome_name;
        this->pos_start = pos_start;
        this->pos_end = pos_end;
        this->depth = depth;
    }
    PositionalCandidateRecord() {
    }
    bool operator< (const PositionalCandidateRecord& that ) const {
        if(this->chromosome_name != that.chromosome_name) return this->chromosome_name < that.chromosome_name;
        if(this->pos_start != that.pos_start) return this->pos_start < that.pos_start;
        if(this->pos_end != that.pos_end) return this->pos_end < that.pos_end;
        return this->depth > that.depth;
    }
};

class CandidateFinder {
    long long region_start;
    long long region_end;
    long long ref_start;
    long long ref_end;
    string chromosome_name;
    string reference_sequence;
    map<Candidate, int> AlleleFrequencyMap;
    map<Candidate, int> AlleleFrequencyMapH0;
    map<Candidate, int> AlleleFrequencyMapH1;
    map<Candidate, int> AlleleFrequencyMapH2;
    vector< set<Candidate> > AlleleMap;
public:
    CandidateFinder(string reference_sequence,
                    string chromosome_name,
                    long long region_start,
                    long long region_end,
                    long long ref_start,
                    long long ref_end);
    void add_read_alleles(type_read &read, vector<int> &coverage);
    vector<PositionalCandidateRecord> find_candidates(vector<type_read>& reads);
    // this is for speed-up, we are going to memorize all position wise read-indicies
    map<long long, set<int> > position_to_read_map;
};



#endif //PEPPER_HP_CANDIDATE_FINDER_H