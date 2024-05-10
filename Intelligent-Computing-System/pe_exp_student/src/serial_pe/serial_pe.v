module serial_pe(
  input                clk,
  input                rst_n,
  input  signed [15:0] neuron,
  input  signed [15:0] weight,
  input         [ 1:0] ctl,
  input                vld_i,
  output        [31:0] result,
  output reg           vld_o
);

/*multiplier*/  /*TODO*/
wire signed [31:0] mult_res = ____________________________________________;
reg [31:0] psum_r;

/*加法器*/  /*TODO*/
wire [31:0] psum_d = _________________________________________________;

/*部分和寄存器*/
always@(posedge clk or negedge rst_n)
if(!rst_n) begin
  psum_r <= 32'h0;
end else if(vld_i) begin
  psum_r <= psum_d;
end

always@(posedge clk or negedge rst_n)
if(!rst_n) begin
  vld_o <= 1'b0;
end else if(____________________________________________________) begin
  vld_o <= 1'b1;
end else begin
  vld_o <= 1'b0;
end

assign result = psum_r;

endmodule
