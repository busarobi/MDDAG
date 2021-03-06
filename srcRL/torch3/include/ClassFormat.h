// Copyright (C) 2003--2004 Ronan Collobert (collober@idiap.ch)
//                
// This file is part of Torch 3.1.
//
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef CLASS_FORMAT_INC
#define CLASS_FORMAT_INC

#include "Object.h"

namespace Torch {

/** Used to define a class code.

    @author Ronan Collobert (collober@idiap.ch)
*/
class ClassFormat : public Object
{
  public:

    /// Number of classes that the object currently handles
    int n_classes;

    /// The label of each class
    real **class_labels;
    
    ///
    ClassFormat();

    /// Returns the output size.
    virtual int getOutputSize() = 0;

    /// Transforms the output from a OneHot representation.
    virtual void fromOneHot(real *outputs, real *one_hot_outputs) = 0;

    /// Transforms the output to a OneHot representation.
    virtual void toOneHot(real *outputs, real *one_hot_outputs) = 0;

    /// Returns the class of #vector#.
    virtual int getClass(real *vector) = 0;

    virtual ~ClassFormat();
};

}

#endif
