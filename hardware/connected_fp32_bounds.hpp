// Same-input connected-C references. Never synthesis or detector-score tolerance.
// Formulas/bounds: fixed_fp32_service_tb.cpp and run_fixed_fp32_rtl.py.
#ifndef SENTRY_CONNECTED_FP32_BOUNDS_HPP
#define SENTRY_CONNECTED_FP32_BOUNDS_HPP
#include <ap_int.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace connected_bounds {
inline std::string json_number(double v) {
    if(!std::isfinite(v))return "\"nonfinite\"";
    std::ostringstream s;s<<std::setprecision(17)<<v;return s.str();
}
static long long checked_calls[9]={0}, checked_values[9]={0}, total_calls=0;
struct Call {
    int op, rows, width, outputs;
    long long index, op_index, values=0, softmax_rows=0;
    double atol=0, rtol=0, maximum_absolute=0, maximum_relative=0;
    double maximum_bound_ratio=0, maximum_row_sum_error=0;
    std::string directory;
    bool passed=true;
    long long first_index=-1;
    double first_actual=0, first_expected=0, first_error=0, first_allowed=0;
    Call(int operation,int r,int w,int n,const std::string& folder)
        :op(operation),rows(r),width(w),outputs(n),directory(folder) {
        if(op<0 || op>8 || rows<=0 || width<=0)throw std::runtime_error("invalid connected bounds call");
        index=++total_calls;op_index=checked_calls[op]+1;
        if(op==0)atol=rtol=3.0e-5;
        else if(op==5)atol=rtol=2.0e-4;
        else if(op>=6){atol=3.0e-6;rtol=3.0e-5;}
    }
    void record() const {
        std::ofstream f(directory+"/operator_bounds.jsonl",std::ios::app);
        f<<std::setprecision(17)<<"{\"call_index\":"<<index<<",\"op\":"<<op
         <<",\"op_call_index\":"<<op_index<<",\"rows\":"<<rows<<",\"width\":"<<width
         <<",\"outputs\":"<<outputs<<",\"values_checked\":"<<values
         <<",\"atol\":"<<atol<<",\"rtol\":"<<rtol
         <<",\"maximum_absolute_error\":"<<maximum_absolute
         <<",\"maximum_relative_error\":"<<maximum_relative
         <<",\"maximum_bound_ratio\":"<<maximum_bound_ratio
         <<",\"softmax_rows_checked\":"<<softmax_rows
         <<",\"maximum_softmax_row_sum_error\":"<<maximum_row_sum_error
         <<",\"softmax_row_sum_atol\":"<<(op==6?"2e-5":"null")
         <<",\"exact_formula_required\":"<<((op>=1&&op<=4)?"true":"false")
         <<",\"passed\":"<<(passed?"true":"false");
        if(!passed)f<<",\"first_violation\":{\"index\":"<<first_index<<",\"actual\":"<<json_number(first_actual)
                   <<",\"expected\":"<<json_number(first_expected)<<",\"absolute_error\":"<<json_number(first_error)
                   <<",\"allowed_error\":"<<json_number(first_allowed)<<"}";
        f<<"}\n";f.flush();
        if(!f)throw std::runtime_error("cannot preserve connected operator bounds");
    }
    void fail(long long i,double actual,double expected,double error,double allowed) {
        passed=false;first_index=i;first_actual=actual;first_expected=expected;
        first_error=error;first_allowed=allowed;record();
        throw std::runtime_error("connected FP32 bound violation op="+std::to_string(op)+
                                 " call="+std::to_string(index)+" index="+std::to_string(i));
    }
    void check(float actual,long double expected,long long i) {
        const double reference=static_cast<double>(expected);
        if(!std::isfinite(actual)||!std::isfinite(reference))fail(i,actual,reference,std::numeric_limits<double>::infinity(),atol+rtol*std::abs(reference));
        const double error=std::abs(double(actual)-reference);
        const double relative=std::abs(reference)>1.0e-30?error/std::abs(reference):0.0;
        const double allowed=atol+rtol*std::abs(reference);
        ++values;maximum_absolute=std::max(maximum_absolute,error);
        maximum_relative=std::max(maximum_relative,relative);
        if(allowed>0)maximum_bound_ratio=std::max(maximum_bound_ratio,error/allowed);
        if(error>allowed)fail(i,actual,reference,error,allowed);
    }
    void finish(){record();++checked_calls[op];checked_values[op]+=values;}
};
inline void extent(const std::vector<float>& values,size_t n) {
    if(values.size()!=n)throw std::runtime_error("connected reference input extent");
}
inline void check(int op,const std::vector<float>& x,const std::vector<float>& y,
                  const std::vector<float>& z,const std::vector<ap_int<32>>& acc,
                  const std::vector<float>& actual,int rows,int width,int outputs,
                  float scale,bool bias,const std::string& directory) {
    Call call(op,rows,width,outputs,directory);
    const int count=rows*width;extent(actual,op==0?rows*outputs:count);
    if(op==0) {
        extent(x,count);extent(y,outputs*width);if(bias)extent(z,outputs);
        for(int r=0;r<rows;++r)for(int o=0;o<outputs;++o) {
            long double expected=bias?z[o]:0.0L;
            for(int k=0;k<width;++k)expected+=static_cast<long double>(x[r*width+k])*y[o*width+k];
            call.check(actual[r*outputs+o],expected,r*outputs+o);
        }
    } else if(op==2) {
        extent(y,width);extent(z,width);
        if(acc.size()!=size_t(count))throw std::runtime_error("connected accumulator extent");
        for(int i=0;i<count;++i) {
            // Independent FP32 conversion, combined scale, product, then bias.
            volatile float converted=static_cast<float>(acc[i].to_int());
            volatile float combined=scale*y[i%width];
            volatile float product=converted*combined;
            volatile float expected=product+z[i%width];call.check(actual[i],expected,i);
        }
    } else if(op==3 || op==4) {
        extent(x,count);extent(y,count);if(op==3)extent(z,count);
        for(int i=0;i<count;++i) {
            volatile float sum=x[i]+y[i];volatile float expected=op==3?sum+z[i]:sum;
            call.check(actual[i],expected,i);
        }
    } else if(op==5) {
        extent(x,count);extent(y,width);extent(z,width);
        for(int r=0;r<rows;++r) {
            long double mean=0.0L,variance=0.0L;
            for(int c=0;c<width;++c)mean+=x[r*width+c];mean/=width;
            for(int c=0;c<width;++c){const long double v=static_cast<long double>(x[r*width+c])-mean;variance+=v*v;}
            variance/=width;
            for(int c=0;c<width;++c)call.check(actual[r*width+c],
                ((static_cast<long double>(x[r*width+c])-mean)/std::sqrt(variance+1.0e-12L))*y[c]+z[c],r*width+c);
        }
    } else if(op==6) {
        extent(x,count);extent(y,count);
        for(int r=0;r<rows;++r) {
            std::vector<long double> scores(width);
            long double maximum=-std::numeric_limits<long double>::max(),sum=0.0L;
            for(int c=0;c<width;++c){scores[c]=static_cast<long double>(x[r*width+c])/scale+y[r*width+c];maximum=std::max(maximum,scores[c]);}
            for(int c=0;c<width;++c){scores[c]=std::exp(scores[c]-maximum);sum+=scores[c];}
            double actual_sum=0.0;
            for(int c=0;c<width;++c){call.check(actual[r*width+c],scores[c]/sum,r*width+c);actual_sum+=actual[r*width+c];}
            ++call.softmax_rows;const double error=std::abs(actual_sum-1.0);
            call.maximum_row_sum_error=std::max(call.maximum_row_sum_error,error);
            if(error>2.0e-5)call.fail(r,actual_sum,1.0,error,2.0e-5);
        }
    } else if(op==7 || op==8) {
        extent(x,count);
        for(int i=0;i<count;++i) {
            const long double v=x[i];
            const long double expected=op==7?0.5L*v*(1.0L+std::erf(v/std::sqrt(2.0L))):std::tanh(v);
            call.check(actual[i],expected,i);
        }
    } else throw std::runtime_error("quantization uses existing same-input exact check");
    call.finish();
}
// Called only after the complete invocation passed canonical independent
// same-input division/nearest-even/clipping comparisons. No arithmetic change.
inline void quantization_failed(int rows,int width,int i,int actual,int expected,const std::string& directory) {
    Call call(1,rows,width,0,directory);call.values=i+1;
    call.maximum_absolute=std::abs(double(actual)-expected);
    call.fail(i,actual,expected,call.maximum_absolute,0.0);
}
inline void quantization_checked(int rows,int width,const std::string& directory) {
    Call call(1,rows,width,0,directory);call.values=rows*width;call.finish();
}
inline void write_summary(std::ostream& f,const long long* actual_calls) {
    for(int op=0;op<9;++op)if(checked_calls[op]!=actual_calls[op])throw std::runtime_error("connected bounds call coverage differs");
    f<<"{\"schema\":\"sentry_connected_fp32_bounds_v2\",\"all_service_calls_checked\":true,\"checked_calls_by_op\":[";
    for(int op=0;op<9;++op)f<<(op?",":"")<<checked_calls[op];
    f<<"],\"checked_values_by_op\":[";for(int op=0;op<9;++op)f<<(op?",":"")<<checked_values[op];
    f<<"],\"reference_mantissa_bits\":"<<std::numeric_limits<long double>::digits<<",\"fp32_mantissa_bits\":"<<std::numeric_limits<float>::digits<<",\"record\":\"operator_bounds.jsonl\",\"reference\":\"Same actual inputs; independent long-double formulas and explicitly rounded FP32 exact formulas; original operator bounds, no detector-score tolerance\"}";
}
} // namespace connected_bounds
#endif
