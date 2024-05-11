module pe_mult(
  input  [ 511:0] mult_neuron,
  input  [ 511:0] mult_weight,
  output [1023:0] mult_result
);

// *******************************************************************
// int16 mult
// *******************************************************************
genvar i;
wire signed [15:0] int16_neuron[31:0];
wire signed [15:0] int16_weight[31:0];
wire signed [31:0] int16_mult_result[31:0];
generate
  for(i=0; i<32; i=i+1)
  begin:int16_mult
    assign int16_neuron[i] = mult_neuron[i*16+15-:16];
    assign int16_weight[i] = mult_weight[i*16+15-:16];
    assign int16_mult_result[i] = int16_neuron[i] * int16_weight[i];
    assign mult_result[i*32+31-:32] = int16_mult_result[i];
  end
endgenerate

endmodule
