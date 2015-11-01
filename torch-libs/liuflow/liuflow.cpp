
// To load this lib in LUA:
// require 'libliuflow'

#include <luaT.h>
#include <TH.h>

#include "project.h"
#include "Image.h"
#include "OpticalFlow.h"
#include <iostream>

using namespace std;

// conversion functions
static DImage *tensor_to_image(THFloatTensor *tensor) {
  // create output
  int w = tensor->size[2];
  int h = tensor->size[1];
  int c = tensor->size[0];
  DImage *img = new DImage(w,h,c);

  // copy data
  int i1,i0,i2;
  double *dest = img->data();
  int offset = 0;
  for (i1=0; i1<tensor->size[1]; i1++) {
    for (i2=0; i2<tensor->size[2]; i2++) {
      for (i0=0; i0<tensor->size[0]; i0++) {
        dest[offset++] = THFloatTensor_get3d(tensor, i0, i1, i2);
      }
    }
  }

  // return result
  return img;
}

static THFloatTensor *image_to_tensor(DImage *img) {
  // create output
  THFloatTensor *tensor = THFloatTensor_newWithSize3d(img->nchannels(), img->height(), img->width());

  // copy data
  int i1,i0,i2;
  double *src = img->data();
  int offset = 0;
  for (i1=0; i1<tensor->size[1]; i1++) {
    for (i2=0; i2<tensor->size[2]; i2++) {
      for (i0=0; i0<tensor->size[0]; i0++) {
        THFloatTensor_set3d(tensor, i0, i1, i2, src[offset++]);
      }
    }
  }

  // return result
  return tensor;
}

int optflow_lua(lua_State *L) {
  // defaults
  double alpha=0.01;
  double ratio=0.75;
  int minWidth=30;
  int nOuterFPIterations=15;
  int nInnerFPIterations=1;
  int nCGIterations=40;

  // get args
  THFloatTensor *ten1 = (THFloatTensor *)luaT_checkudata(L, 1, luaT_checktypename2id(L, "torch.FloatTensor"));
  THFloatTensor *ten2 = (THFloatTensor *)luaT_checkudata(L, 2, luaT_checktypename2id(L, "torch.FloatTensor"));
  if (lua_isnumber(L, 3)) alpha = lua_tonumber(L, 3);
  if (lua_isnumber(L, 4)) ratio = lua_tonumber(L, 4);
  if (lua_isnumber(L, 5)) minWidth = lua_tonumber(L, 5);
  if (lua_isnumber(L, 6)) nOuterFPIterations = lua_tonumber(L, 6);
  if (lua_isnumber(L, 7)) nInnerFPIterations = lua_tonumber(L, 7);
  if (lua_isnumber(L, 8)) nCGIterations = lua_tonumber(L, 8);

  // copy tensors to images
  DImage *img1 = tensor_to_image(ten1);
  DImage *img2 = tensor_to_image(ten2);

  // declare outputs, and process
  DImage vx,vy,warpI2;
  OpticalFlow::Coarse2FineFlow(vx,vy,warpI2,         // outputs
                               *img1,*img2,          // inputs
                               alpha,ratio,minWidth, // params
                               nOuterFPIterations,nInnerFPIterations,nCGIterations);

  // return result
  THFloatTensor *ten_vx = image_to_tensor(&vx);
  THFloatTensor *ten_vy = image_to_tensor(&vy);
  THFloatTensor *ten_warp = image_to_tensor(&warpI2);
  luaT_pushudata(L, ten_vx, luaT_checktypename2id(L, "torch.FloatTensor"));
  luaT_pushudata(L, ten_vy, luaT_checktypename2id(L, "torch.FloatTensor"));
  luaT_pushudata(L, ten_warp, luaT_checktypename2id(L, "torch.FloatTensor"));

  // cleanup
  delete(img1);
  delete(img2);

  return 3;
}

int warp_lua(lua_State *L) {
  // get args
  THFloatTensor *ten_inp = (THFloatTensor *)luaT_checkudata(L, 1, luaT_checktypename2id(L, "torch.Tensor"));
  THFloatTensor *ten_vx = (THFloatTensor *)luaT_checkudata(L, 2, luaT_checktypename2id(L, "torch.Tensor"));
  THFloatTensor *ten_vy = (THFloatTensor *)luaT_checkudata(L, 3, luaT_checktypename2id(L, "torch.Tensor"));

  // copy tensors to images
  DImage *input = tensor_to_image(ten_inp);
  DImage *vx = tensor_to_image(ten_vx);
  DImage *vy = tensor_to_image(ten_vy);

  // declare outputs, and process
  DImage warpedInput;
  OpticalFlow::warpFL(warpedInput,   // warped input
                      *input,*input, // input
                      *vx, *vy         // flow
                      );

  // return result
  THFloatTensor *ten_warp = image_to_tensor(&warpedInput);
  luaT_pushudata(L, ten_warp, luaT_checktypename2id(L, "torch.Tensor"));

  // cleanup
  delete(input);
  delete(vx);
  delete(vy);

  return 1;
}

// Register functions in LUA
static const struct luaL_reg liuflow [] = {
  {"infer", optflow_lua},
  {"warp", warp_lua},
  {NULL, NULL}  /* sentinel */
};

extern "C" {
  int luaopen_libliuflow (lua_State *L) {
    luaL_openlib(L, "libliuflow", liuflow, 0);
    return 1; 
  }
}
