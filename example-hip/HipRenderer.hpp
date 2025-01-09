#pragma once

#include <hip/hip_runtime.h>

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>


class HipRenderer final : public OSL::RendererServices
{

public:
    HipRenderer() = default;
    virtual ~HipRenderer() = default;

    virtual int supports(OIIO::string_view feature) const override;
};
