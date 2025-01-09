#include "HipRenderer.hpp"


int HipRenderer::supports(OIIO::string_view feature) const
{
    if (feature == "HIP") 
    {
        return true;
    }
    return false;
}
