
function id(x)
    --@generic T
    --@param x: T
    if x then return x end
    return nil
end

local x = id(10)

return x
