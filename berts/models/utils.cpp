#include "berts/models/utils.hpp"
#include <type_traits>

namespace berts {

template <typename Ctx>
concept uniq = requires(Ctx ctx) {
    requires std::is_move_constructible_v<Ctx>;
    requires std::is_nothrow_move_constructible_v<Ctx>;
    requires std::is_move_assignable_v<Ctx>;
    requires std::is_nothrow_move_assignable_v<Ctx>;
    requires !std::is_copy_constructible_v<Ctx>;
    requires !std::is_copy_assignable_v<Ctx>;
};

static_assert(uniq<berts_ctx>);
static_assert(uniq<ggml_ctx>);
static_assert(uniq<gguf_ctx>);
static_assert(uniq<gg_ctx>);

} // namespace berts
