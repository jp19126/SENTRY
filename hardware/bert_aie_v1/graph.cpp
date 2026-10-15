#include "tiles.hpp"
using namespace adf;
class BertGraph:public graph {
public:
    input_plio ia,ib,fa,fb;output_plio ic,fc;kernel ik,fk;
    BertGraph() {
        ia=input_plio::create("i8_a",plio_128_bits,"data/i8_a.txt",250);
        ib=input_plio::create("i8_b",plio_128_bits,"data/i8_b.txt",250);
        ic=output_plio::create("i8_c",plio_128_bits,"data/i8_c.txt",250);
        fa=input_plio::create("bf_a",plio_128_bits,"data/bf_a.txt",250);
        fb=input_plio::create("bf_b",plio_128_bits,"data/bf_b.txt",250);
        fc=output_plio::create("bf_c",plio_128_bits,"data/bf_c.txt",250);
        ik=kernel::create(int8_tile);fk=kernel::create(bf16_tile);
        source(ik)="tiles.cpp";source(fk)="tiles.cpp";
        runtime<ratio>(ik)=0.9;runtime<ratio>(fk)=0.9;
        location<kernel>(ik)=tile(10,0);location<kernel>(fk)=tile(11,0);
        connect(ia.out[0],ik.in[0]);connect(ib.out[0],ik.in[1]);connect(ik.out[0],ic.in[0]);
        connect(fa.out[0],fk.in[0]);connect(fb.out[0],fk.in[1]);connect(fk.out[0],fc.in[0]);
        dimensions(ik.in[0])={8192};dimensions(ik.in[1])={8192};dimensions(ik.out[0])={1024};
        dimensions(fk.in[0])={2048};dimensions(fk.in[1])={2048};dimensions(fk.out[0])={1024};
    }
};
BertGraph bert_graph;
#if defined(__AIESIM__) || defined(__X86SIM__)
int main(){bert_graph.init();bert_graph.run(2);bert_graph.end();return 0;}
#endif
